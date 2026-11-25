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
    in += blockIdx.y * width;
    out += blockIdx.y * width;
    __shared__ float scratch[256];
    int lane = threadIdx.x;
    float total = 0;
    for (int i = lane; i < width; i += blockDim.x) total += in[i] * in[i];
    float scale = rsqrtf(sum_block(total, scratch) / width + eps);
    for (int i = lane; i < width; i += blockDim.x) out[i] = in[i] * scale * w[i];
}

extern "C" __global__ void residual(float* x, const float* y, int width) {
    x += blockIdx.y * width;
    y += blockIdx.y * width;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) x[i] += y[i];
}

extern "C" __global__ void swiglu(float* gate, const float* up, int width) {
    gate += blockIdx.y * width;
    up += blockIdx.y * width;
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
                                       int position, float base, float eps, const int* cursor) {
    if (cursor) position = *cursor;
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
                                      float* values, int position, const int* cursor) {
    if (cursor) position = *cursor;
    int i = blockIdx.x * 256 + threadIdx.x;
    keys[position * 512 + i] = k[i];
    values[position * 512 + i] = v[i];
}

extern "C" __global__ void scores(const float* q, const float* keys, float* score,
                                    int length, int stride, const int* cursor) {
    if (cursor) length = *cursor + 1;
    __shared__ float scratch[256];
    int h = blockIdx.y, t = blockIdx.x, i = threadIdx.x;
    if (t >= length) return;
    float product = q[h * 512 + i] * keys[t * 512 + (h / 4) * 256 + i];
    float result = sum_block(product, scratch) / 16.f;
    if (i == 0) score[h * stride + t] = result;
}

extern "C" __global__ void read_attention(const float* q, const float* values,
                                            const float* score, float* out, int length,
                                            int stride, const int* cursor) {
    if (cursor) length = *cursor + 1;
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

// Packed rows follow GGUF blocks. Each warp owns one output row, and four
// independent warps share a block. Thirty-two contiguous input lanes share
// K-quant scale metadata and avoid writing a dequantized matrix.
__device__ float block_half(const unsigned char* p) {
    unsigned short bits = p[0] | (p[1] << 8);
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ float packed_value(const unsigned char* row, int pos, int type) {
    if (type == 8) {
        const auto* p = row + (pos / 32) * 34;
        return block_half(p) * (signed char)p[2 + pos % 32];
    }
    int item = pos % 256;
    if (type == 14) {
        const auto* p = row + (pos / 256) * 210;
        int section = item / 128, quarter = item / 32 % 4, lane = item % 32;
        int low = (p[section * 64 + quarter % 2 * 32 + lane] >> (4 * (quarter / 2))) & 15;
        int high = (p[128 + section * 32 + lane] >> (2 * quarter)) & 3;
        return block_half(p + 208) * (signed char)p[192 + item / 16] *
            ((low | (high << 4)) - 32);
    }
    const bool fifth = type == 13;
    const auto* p = row + (pos / 256) * (fifth ? 176 : 144);
    const auto* scale = p + 4;
    int group = item / 32, lane = item % 32;
    int ds = group < 4 ? (scale[group] & 63)
                       : ((scale[group + 4] & 15) | ((scale[group - 4] >> 6) << 4));
    int ms = group < 4 ? (scale[group + 4] & 63)
                       : ((scale[group + 4] >> 4) | ((scale[group] >> 6) << 4));
    const auto* bits = p + (fifth ? 48 : 16);
    int q = (bits[group / 2 * 32 + lane] >> (4 * (group % 2))) & 15;
    if (fifth) q |= ((p[16 + lane] >> group) & 1) << 4;
    return block_half(p) * ds * q - block_half(p + 2) * ms;
}

extern "C" __global__ void packed_matvec(const unsigned char* data, const float* x,
                                         float* y, int cols, int rows, int bytes, int type) {
    int lane = threadIdx.x % 32, r = blockIdx.x * 4 + threadIdx.x / 32;
    if (r >= rows) return;
    const auto* row = data + static_cast<unsigned long long>(r) * bytes;
    float sum = 0;
    for (int i = lane; i < cols; i += 32) sum += packed_value(row, i, type) * x[i];
    for (int i = 16; i; i /= 2) sum += __shfl_down_sync(0xffffffff, sum, i);
    if (lane == 0) y[r] = sum;
}

// Four warps compute four Q6_K output rows. Within a 256-weight block,
// one lane owns one byte from each 32-byte low/high stripe. Reuse those
// bytes for four consecutive products while retaining FP32 sum order.
extern "C" __global__ void q6_block_matvec(const unsigned char* data, const float* x,
                                            float* y, int cols, int rows, int bytes) {
    const int lane = threadIdx.x % 32;
    const int r = blockIdx.x * 4 + threadIdx.x / 32;
    if (r >= rows) return;
    const auto* row = data + static_cast<unsigned long long>(r) * bytes;
    float sum = 0;
    for (int block = 0; block < cols / 256; ++block) {
        const auto* p = row + block * 210;
        const float d = block_half(p + 208);
        #pragma unroll
        for (int section = 0; section < 2; ++section) {
            const int high = p[128 + section * 32 + lane];
            const int low[2] = {p[section * 64 + lane], p[section * 64 + 32 + lane]};
            #pragma unroll
            for (int quarter = 0; quarter < 4; ++quarter) {
                const int scale = (signed char)p[192 + section * 8 + quarter * 2 + lane / 16];
                const int ql = (low[quarter % 2] >> (4 * (quarter / 2))) & 15;
                const int qh = (high >> (2 * quarter)) & 3;
                const int q = (ql | (qh << 4)) - 32;
                sum += (d * scale * q) * x[block * 256 + section * 128 + quarter * 32 + lane];
            }
        }
    }
    for (int i = 16; i; i /= 2) sum += __shfl_down_sync(0xffffffff, sum, i);
    if (lane == 0) y[r] = sum;
}

// One 256-lane block writes a contiguous stripe of an unpacked matrix row.
extern "C" __global__ void unpack_matrix(const unsigned char* data, float* output,
                                          int cols, int bytes, int type) {
    const int row = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) {
        const auto* start = data + static_cast<unsigned long long>(row) * bytes;
        output[row * cols + i] = packed_value(start, i, type);
    }
}

__device__ unsigned short fp16_bits(float value) {
    unsigned short bits;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(value));
    return bits;
}

// 256 lanes convert contiguous packed-matrix stripes and activation vectors
// for prompt GEMM. Quantized storage outside this scratch path stays packed.
extern "C" __global__ void unpack_matrix_half(const unsigned char* data, unsigned short* output,
                                               int cols, int bytes, int type) {
    const int row = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) {
        const auto* start = data + static_cast<unsigned long long>(row) * bytes;
        output[row * cols + i] = fp16_bits(packed_value(start, i, type));
    }
}

extern "C" __global__ void to_half(const float* in, unsigned short* output, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = fp16_bits(in[i]);
}

extern "C" __global__ void packed_embedding(const unsigned char* row, float* x, int type) {
    int i = blockIdx.x * 256 + threadIdx.x;
    x[i] = packed_value(row, i, type);
}
)CUDA"
