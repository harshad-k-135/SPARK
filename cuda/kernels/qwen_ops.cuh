// Qwen 0.8B single-token CUDA operations. Resident vectors and matrix rows
// are contiguous. Reductions use 128 or 256 lanes to cover the native heads;
// recurrent state is [head][key channel][value channel].
R"CUDA(
__device__ float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
__device__ float silu(float x) { return x * sigmoid(x); }

__device__ float sum_block(float value, float* scratch) {
    int i = threadIdx.x;
    scratch[i] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (i < stride) scratch[i] += scratch[i + stride];
        __syncthreads();
    }
    float total = scratch[0];
    __syncthreads();
    return total;
}

extern "C" __global__ void rms(const float* in, const float* w, float* out,
                                 int width, float eps) {
    __shared__ float scratch[256];
    int lane = threadIdx.x;
    float total = 0;
    for (int i = lane; i < width; i += blockDim.x) total += in[i] * in[i];
    float scale = rsqrtf(sum_block(total, scratch) / width + eps);
    for (int i = lane; i < width; i += blockDim.x) out[i] = in[i] * scale * w[i];
}

extern "C" __global__ void residual(float* x, const float* y, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) x[i] += y[i];
}

extern "C" __global__ void swiglu(float* gate, const float* up, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) gate[i] = silu(gate[i]) * up[i];
}

extern "C" __global__ void convolve(float* qkv, float* history, const float* weights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float* h = history + 3 * i;
    const float* w = weights + 4 * i;
    float x = qkv[i];
    qkv[i] = silu(h[0] * w[0] + h[1] * w[1] + h[2] * w[2] + x * w[3]);
    h[0] = h[1]; h[1] = h[2]; h[2] = x;
}

extern "C" __global__ void delta(const float* qkv, const float* z, const float* alpha,
                                   const float* beta, const float* a, const float* bias,
                                   const float* weights, float* states, float* out, float eps) {
    __shared__ float scratch[128];
    __shared__ float normalized_k[128];
    __shared__ float normalized_q[128];
    int h = blockIdx.x, i = threadIdx.x;
    const float* q = qkv + h * 128;
    const float* k = q + 2048;
    const float* v = k + 2048;
    float* state = states + h * 128 * 128;
    const float q_norm = sum_block(q[i] * q[i], scratch);
    const float k_norm = sum_block(k[i] * k[i], scratch);
    normalized_q[i] = q[i] * rsqrtf(q_norm + eps) * rsqrtf(128.f);
    normalized_k[i] = k[i] * rsqrtf(k_norm + eps);
    __syncthreads();
    const float x = alpha[h] + bias[h];
    const float decay = expf(a[h] * (fmaxf(0.f, x) + log1pf(expf(-fabsf(x)))));
    float previous = 0;
    for (int j = 0; j < 128; ++j) {
        float decayed = state[j * 128 + i] * decay;
        state[j * 128 + i] = decayed;
        previous += normalized_k[j] * decayed;
    }
    float change = (v[i] - previous) * sigmoid(beta[h]);
    float result = 0;
    for (int j = 0; j < 128; ++j) {
        float next = state[j * 128 + i] + normalized_k[j] * change;
        state[j * 128 + i] = next;
        result += normalized_q[j] * next;
    }
    const float rms_value = sum_block(result * result, scratch);
    out[h * 128 + i] = result * rsqrtf(rms_value / 128.f + eps)
                       * weights[i] * silu(z[h * 128 + i]);
}

extern "C" __global__ void norm_rope(float* values, const float* w, int stride,
                                       int position, float base, float eps) {
    __shared__ float scratch[256];
    __shared__ float rotated[256];
    int i = threadIdx.x;
    float* head = values + blockIdx.x * stride;
    const float x = head[i];
    const float scale = rsqrtf(sum_block(x * x, scratch) / 256.f + eps);
    rotated[i] = x * scale * w[i];
    __syncthreads();
    if (i < 32) {
        const float theta = position / powf(base, i / 32.f);
        const float c = cosf(theta), s = sinf(theta);
        head[i] = rotated[i] * c - rotated[i + 32] * s;
        head[i + 32] = rotated[i + 32] * c + rotated[i] * s;
    } else if (i >= 64) head[i] = rotated[i];
}

extern "C" __global__ void cache_kv(const float* k, const float* v, float* keys,
                                      float* values, int position) {
    int i = blockIdx.x * 256 + threadIdx.x;
    keys[position * 512 + i] = k[i];
    values[position * 512 + i] = v[i];
}

extern "C" __global__ void scores(const float* q, const float* keys, float* score,
                                    int length, int stride) {
    __shared__ float scratch[256];
    int h = blockIdx.y, t = blockIdx.x, i = threadIdx.x;
    if (t >= length) return;
    float product = q[h * 512 + i] * keys[t * 512 + (h / 4) * 256 + i];
    float result = sum_block(product, scratch) / 16.f;
    if (i == 0) score[h * stride + t] = result;
}

extern "C" __global__ void read_attention(const float* q, const float* values,
                                            const float* score, float* out, int length,
                                            int stride) {
    __shared__ float scratch[256];
    int h = blockIdx.x, i = threadIdx.x;
    const float* scores = score + h * stride;
    float highest = -3.402823466e+38F;
    for (int t = i; t < length; t += 256) highest = fmaxf(highest, scores[t]);
    scratch[i] = highest;
    __syncthreads();
    for (int half = 128; half; half /= 2) {
        if (i < half) scratch[i] = fmaxf(scratch[i], scratch[i + half]);
        __syncthreads();
    }
    highest = scratch[0];
    __syncthreads();
    float mass = 0;
    for (int t = i; t < length; t += 256) mass += expf(scores[t] - highest);
    mass = sum_block(mass, scratch);
    float total = 0;
    for (int t = 0; t < length; ++t)
        total += expf(scores[t] - highest) * values[t * 512 + (h / 4) * 256 + i];
    out[h * 256 + i] = total / mass * sigmoid(q[h * 512 + 256 + i]);
}
)CUDA"
