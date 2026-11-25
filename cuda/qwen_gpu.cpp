#include "cuda/qwen_gpu.hpp"
#include "cuda/check.hpp"
#include "cuda/memory/arena.hpp"
#include "engine/model_loader/packed_weights.hpp"
#include "engine/transformer/delta_net.hpp"

#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>

namespace spark {
namespace {

#ifndef SPARK_COMPUTE_ARCH
#define SPARK_COMPUTE_ARCH "86"
#endif

constexpr char source[] =
#include "cuda/kernels/qwen_ops.cuh"
;

struct Context {
    CUdevice device = 0;
    CUcontext value = nullptr;
    Context() {
        SPARK_CUDA(cuInit(0));
        SPARK_CUDA(cuDeviceGet(&device, 0));
        SPARK_CUDA(cuDevicePrimaryCtxRetain(&value, device));
        try {
            SPARK_CUDA(cuCtxSetCurrent(value));
        } catch (...) {
            SPARK_CUDA_CLEANUP(cuDevicePrimaryCtxRelease(device));
            throw;
        }
    }
    ~Context() { SPARK_CUDA_CLEANUP(cuDevicePrimaryCtxRelease(device)); }
};

struct Stream {
    CUstream value = nullptr;
    Stream() { SPARK_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~Stream() { SPARK_CUDA_CLEANUP(cuStreamDestroy(value)); }
};

struct Blas {
    cublasHandle_t value = nullptr;
    explicit Blas(CUstream stream) {
        SPARK_CUDA(cublasCreate(&value));
        try {
            SPARK_CUDA(cublasSetStream(value, stream));
            SPARK_CUDA(cublasSetMathMode(value, CUBLAS_PEDANTIC_MATH));
        } catch (...) {
            SPARK_CUDA_CLEANUP(cublasDestroy(value));
            throw;
        }
    }
    ~Blas() { SPARK_CUDA_CLEANUP(cublasDestroy(value)); }
};

struct Program {
    nvrtcProgram value = nullptr;
    Program() { SPARK_CUDA(nvrtcCreateProgram(&value, source, "qwen_ops.cu", 0, nullptr, nullptr)); }
    ~Program() { SPARK_CUDA_CLEANUP(nvrtcDestroyProgram(&value)); }
};

struct Module {
    CUmodule value = nullptr;
    Module() {
        Program program;
        const char* flags[] = {"--std=c++20", "--gpu-architecture=compute_" SPARK_COMPUTE_ARCH,
                               "--fmad=false"};
        const auto result = nvrtcCompileProgram(program.value, 3, flags);
        if (result != NVRTC_SUCCESS) {
            std::size_t bytes = 0;
            SPARK_CUDA(nvrtcGetProgramLogSize(program.value, &bytes));
            std::string log(bytes, '\0');
            SPARK_CUDA(nvrtcGetProgramLog(program.value, log.data()));
            std::cerr << log << '\n';
            SPARK_CUDA(result);
        }
        std::size_t bytes = 0;
        SPARK_CUDA(nvrtcGetPTXSize(program.value, &bytes));
        std::vector<char> code(bytes);
        SPARK_CUDA(nvrtcGetPTX(program.value, code.data()));
        SPARK_CUDA(cuModuleLoadData(&value, code.data()));
    }
    ~Module() { SPARK_CUDA_CLEANUP(cuModuleUnload(value)); }
    CUfunction get(const char* name) const {
        CUfunction fn = nullptr;
        SPARK_CUDA(cuModuleGetFunction(&fn, value, name));
        return fn;
    }
};

struct Pinned {
    float* value = nullptr;
    std::size_t size = 0;
    void allocate(std::size_t count) {
        void* memory = nullptr;
        SPARK_CUDA(cuMemAllocHost(&memory, count * sizeof(float)));
        value = static_cast<float*>(memory);
        size = count;
    }
    ~Pinned() { if (value) SPARK_CUDA_CLEANUP(cuMemFreeHost(value)); }
};

struct Profile {
    // Each layer ends its attention phase and its MLP phase with a marker.
    std::array<CUevent, 50> markers{};
    explicit Profile(bool enabled) {
        if (!enabled) return;
        try {
            for (auto& event : markers) SPARK_CUDA(cuEventCreate(&event, CU_EVENT_DEFAULT));
        } catch (...) {
            release();
            throw;
        }
    }
    void release() noexcept {
        for (auto event : markers) if (event) SPARK_CUDA_CLEANUP(cuEventDestroy(event));
    }
    ~Profile() { release(); }
    bool enabled() const { return markers.front() != nullptr; }
    void mark(std::size_t index, CUstream stream) {
        SPARK_CUDA(cuEventRecord(markers[index], stream));
    }
    float elapsed(std::size_t first, std::size_t last) const {
        float milliseconds = 0;
        SPARK_CUDA(cuEventElapsedTime(&milliseconds, markers[first], markers[last]));
        return milliseconds;
    }
    void report() const {
        float attention = 0, recurrent = 0, mlp = 0;
        for (std::size_t layer = 0; layer < 24; ++layer) {
            const auto seconds = elapsed(2 * layer, 2 * layer + 1);
            if ((layer + 1) % 4 == 0) attention += seconds;
            else recurrent += seconds;
            mlp += elapsed(2 * layer + 1, 2 * layer + 2);
        }
        std::cerr << "{\"event\":\"cuda_decode_profile\",\"recurrent_ms\":" << recurrent
                  << ",\"attention_ms\":" << attention << ",\"mlp_ms\":" << mlp
                  << ",\"output_ms\":" << elapsed(48, 49) << "}\n";
    }
};

struct Graphs {
    std::array<CUgraphExec, 14> entries{};
    ~Graphs() {
        for (auto exec : entries) if (exec) SPARK_CUDA_CLEANUP(cuGraphExecDestroy(exec));
    }
};

struct Matrix {
    CUdeviceptr data = 0;
    int cols = 0, rows = 0, bytes = 0, type = 0;
};
struct Layer {
    Matrix norm, post, up, gate, down, q, k, v, out, qnorm, knorm;
    Matrix convolve, z, alpha, beta, a, bias, rnorm;
    CUdeviceptr state = 0, history = 0, keys = 0, values = 0;
};

struct Kernels {
    CUfunction rms, residual, swiglu, convolve, delta, norm_rope, cache_kv, scores, read_attention;
    CUfunction packed_matvec, packed_embedding, unpack_matrix;
    explicit Kernels(const Module& module)
        : rms(module.get("rms")), residual(module.get("residual")), swiglu(module.get("swiglu")),
          convolve(module.get("convolve")), delta(module.get("delta")),
          norm_rope(module.get("norm_rope")), cache_kv(module.get("cache_kv")),
          scores(module.get("scores")), read_attention(module.get("read_attention")),
          packed_matvec(module.get("packed_matvec")), packed_embedding(module.get("packed_embedding")),
          unpack_matrix(module.get("unpack_matrix")) {}
};

constexpr int hidden = 1024, width = 3584, vocab = 248320;
constexpr int prefill_chunk_size = 128;
constexpr int max_layer_matrix = 6144 * hidden;
constexpr int prefill_channels = 3 * hidden + 6144 + 2 * 512 + 2 * 2048 + 2 * width + 2 * 16;

struct PrefillBuffers {
    CUdeviceptr x, n, p, q, k, v, y, gate, up, z, alpha, beta, weights;
};

void validate(const Gguf& gguf, std::size_t context) {
    if (!context || context > 4096) throw std::invalid_argument("invalid CUDA context length");
    if (std::get<std::string>(gguf.metadata("general.architecture")) != "qwen35" ||
        std::get<GgufArray>(gguf.metadata("tokenizer.ggml.tokens")).count != vocab ||
        gguf.tensors().contains("output.weight"))
        throw std::runtime_error("CUDA backend expects tied Qwen3.5-0.8B output weights");
    const std::pair<std::string_view, std::uint64_t> expected[] = {
        {"embedding_length", hidden}, {"feed_forward_length", width}, {"attention.head_count", 8},
        {"attention.head_count_kv", 2}, {"attention.key_length", 256}, {"attention.value_length", 256},
        {"rope.dimension_count", 64}, {"ssm.group_count", 16}, {"ssm.time_step_rank", 16},
        {"ssm.state_size", 128}, {"ssm.inner_size", 2048}, {"ssm.conv_kernel", 4},
        {"block_count", 25}, {"nextn_predict_layers", 1}, {"full_attention_interval", 4},
    };
    for (auto [key, size] : expected)
        if (std::get<std::uint64_t>(gguf.metadata("qwen35." + std::string(key))) != size)
            throw std::runtime_error("unsupported Qwen CUDA dimension " + std::string(key));
}

}  // namespace

struct GpuModel::Impl {
    Context primary;
    Stream stream;
    Blas blas{stream.value};
    Module module;
    Kernels kernel{module};
    Profile profile;
    std::unique_ptr<DeviceArena> arena;
    Pinned host;
    Pinned host_cursor;
    Graphs graphs;
    std::array<Layer, 24> layers;
    Matrix embedding, last_norm;
    CUdeviceptr x, n, p, q, k, v, y, gate, up, z, alpha, beta, scores, logits;
    int length = 0;
    int capacity = 0;
    int emitted = 0;
    DecodeGraph graph_mode;
    PrefillBuffers prefill_buffers{};
    CUdeviceptr cursor = 0;
    float eps = 0, base = 0;

    Impl(const Gguf& gguf, const std::filesystem::path& path, std::size_t context, bool trace,
         DecodeGraph mode, bool packed, bool batch_prefill)
        : profile(trace), capacity(static_cast<int>(context)), graph_mode(mode) {
        validate(gguf, context);
        eps = static_cast<float>(std::get<double>(gguf.metadata("qwen35.attention.layer_norm_rms_epsilon")));
        base = static_cast<float>(std::get<double>(gguf.metadata("qwen35.rope.freq_base")));
        auto budget = static_cast<std::size_t>(6) * context * 512 * 2 * sizeof(float) +
                      static_cast<std::size_t>(18) * (16 * 128 * 128 + 3 * 6144) * sizeof(float) +
                      8 * 1024 * 1024;
        if (batch_prefill) {
            budget += (static_cast<std::size_t>(prefill_channels) * prefill_chunk_size +
                       max_layer_matrix) * sizeof(float) + 13 * 256;
        }
        for (const auto& [name, t] : gguf.tensors()) {
            if (name.starts_with("blk.24.")) continue;
            const auto enc = gguf_encoding(t.type);
            const auto elements = t.bytes / enc.bytes * enc.elements;
            if (elements > (std::numeric_limits<std::size_t>::max() - budget - 256) / sizeof(float))
                throw std::runtime_error("model exceeds CUDA address space");
            budget += (packed ? t.bytes : elements * sizeof(float)) + 256;
        }
        arena = std::make_unique<DeviceArena>(budget);
        std::ifstream file(path, std::ios::binary);
        const bool use_packed = packed;
        const auto load = [&](const std::string& name, int columns, int rows = 1) {
            const auto& t = gguf.tensors().at(name);
            if (t.rank > 2 || t.shape[0] != static_cast<std::uint64_t>(columns) ||
                t.shape[1] != static_cast<std::uint64_t>(rows))
                throw std::runtime_error("invalid CUDA weight shape " + name);
            std::vector<std::uint8_t> packed(static_cast<std::size_t>(t.bytes));
            file.seekg(static_cast<std::streamoff>(t.file_offset));
            if (!file.read(reinterpret_cast<char*>(packed.data()), packed.size()))
                throw std::runtime_error("cannot read " + name);
            if (use_packed) {
                if (t.type != 0 && t.type != 8 && t.type != 12 && t.type != 13 && t.type != 14)
                    throw std::runtime_error("unsupported GPU packed type for " + name);
                const auto gpu = arena->allocate(packed.size());
                SPARK_CUDA(cuMemcpyHtoD(gpu, packed.data(), packed.size()));
                return Matrix{gpu, columns, rows, static_cast<int>(packed.size() / rows),
                              static_cast<int>(t.type)};
            }
            std::vector<float> values(static_cast<std::size_t>(columns) * rows);
            const auto row_bytes = packed.size() / rows;
#ifdef _OPENMP
            #pragma omp parallel for if(rows * columns >= (1 << 19)) schedule(static)
#endif
            for (int row = 0; row < rows; ++row)
                unpack_gguf_row(t.type, std::span(packed).subspan(row * row_bytes, row_bytes),
                                std::span(values).subspan(static_cast<std::size_t>(row) * columns, columns));
            const auto gpu = buffer(values.size());
            SPARK_CUDA(cuMemcpyHtoD(gpu, values.data(), values.size() * sizeof(float)));
            return Matrix{gpu, columns, rows, columns * static_cast<int>(sizeof(float)), 0};
        };
        embedding = load("token_embd.weight", hidden, vocab);
        last_norm = load("output_norm.weight", hidden);
        for (std::size_t i = 0; i < layers.size(); ++i) {
            auto& l = layers[i];
            const auto prefix = "blk." + std::to_string(i) + ".";
            const auto m = [&](const char* suffix, int cols, int rows = 1) {
                return load(prefix + suffix, cols, rows);
            };
            l.norm = m("attn_norm.weight", hidden);
            l.post = m("post_attention_norm.weight", hidden);
            l.up = m("ffn_up.weight", hidden, width);
            l.gate = m("ffn_gate.weight", hidden, width);
            l.down = m("ffn_down.weight", width, hidden);
            if ((i + 1) % 4 == 0) {
                l.q = m("attn_q.weight", hidden, 4096);
                l.k = m("attn_k.weight", hidden, 512);
                l.v = m("attn_v.weight", hidden, 512);
                l.out = m("attn_output.weight", 2048, hidden);
                l.qnorm = m("attn_q_norm.weight", 256);
                l.knorm = m("attn_k_norm.weight", 256);
                l.keys = buffer(context * 512);
                l.values = buffer(context * 512);
            } else {
                l.q = m("attn_qkv.weight", hidden, 6144);
                l.z = m("attn_gate.weight", hidden, 2048);
                l.alpha = m("ssm_alpha.weight", hidden, 16);
                l.beta = m("ssm_beta.weight", hidden, 16);
                l.a = m("ssm_a", 16);
                l.bias = m("ssm_dt.bias", 16);
                l.rnorm = m("ssm_norm.weight", 128);
                l.convolve = m("ssm_conv1d.weight", 4, 6144);
                l.out = m("ssm_out.weight", 2048, hidden);
                l.state = buffer(16 * 128 * 128);
                l.history = buffer(3 * 6144);
                SPARK_CUDA(cuMemsetD32(l.state, 0, 16 * 128 * 128));
                SPARK_CUDA(cuMemsetD32(l.history, 0, 3 * 6144));
            }
        }
        x = buffer(hidden); n = buffer(hidden); p = buffer(hidden);
        q = buffer(6144); k = buffer(512); v = buffer(512); y = buffer(2048);
        gate = buffer(width); up = buffer(width); z = buffer(2048);
        alpha = buffer(16); beta = buffer(16); scores = buffer(context * 8);
        logits = buffer(vocab);
        host.allocate(vocab);
        if (batch_prefill) {
            const auto chunk = [&](int channels) { return buffer(prefill_chunk_size * channels); };
            prefill_buffers = {chunk(hidden), chunk(hidden), chunk(hidden), chunk(6144),
                               chunk(512), chunk(512), chunk(2048), chunk(width), chunk(width),
                               chunk(2048), chunk(16), chunk(16), buffer(max_layer_matrix)};
        }
        if (graph_mode != DecodeGraph::off) {
            static_assert(sizeof(float) == sizeof(int));
            host_cursor.allocate(1);
            cursor = arena->allocate(sizeof(int));
        }
        std::cerr << "{\"event\":\"cuda_plan\",\"backend\":\"" << (packed ? "packed" : "fp32")
                  << "\",\"arena_bytes\":"
                  << budget << ",\"context\":" << capacity << "}\n";
    }

    CUdeviceptr buffer(std::size_t count) { return arena->allocate(count * sizeof(float)); }

    template <typename... Args>
    void launch(CUfunction function, unsigned gx, unsigned gy, unsigned block, Args... args) {
        void* values[] = {static_cast<void*>(&args)...};
        SPARK_CUDA(cuLaunchKernel(function, gx, gy, 1, block, 1, 1, 0,
                                  stream.value, values, nullptr));
    }

    void mv(const Matrix& m, CUdeviceptr in, CUdeviceptr out) {
        if (m.type != 0) {
            launch(kernel.packed_matvec, (m.rows + 3) / 4, 1, 128, m.data, in, out,
                   m.cols, m.rows, m.bytes, m.type);
            return;
        }
        const float one = 1, zero = 0;
        SPARK_CUDA(cublasSgemv(blas.value, CUBLAS_OP_T, m.cols, m.rows,
                               &one, reinterpret_cast<const float*>(m.data), m.cols,
                               reinterpret_cast<const float*>(in), 1, &zero,
                               reinterpret_cast<float*>(out), 1));
    }

    void reset() {
        for (const auto& layer : layers) {
            if (!layer.state) continue;
            SPARK_CUDA(cuMemsetD32Async(layer.state, 0, 16 * 128 * 128, stream.value));
            SPARK_CUDA(cuMemsetD32Async(layer.history, 0, 3 * 6144, stream.value));
        }
        SPARK_CUDA(cuStreamSynchronize(stream.value));
        length = 0;
        emitted = 0;
    }

    void gather(TokenId token, CUdeviceptr out) {
        const auto row = embedding.data + static_cast<std::size_t>(token) * embedding.bytes;
        if (embedding.type != 0) launch(kernel.packed_embedding, 4, 1, 256, row, out, embedding.type);
        else SPARK_CUDA(cuMemcpyDtoDAsync(out, row, hidden * sizeof(float), stream.value));
    }

    void attention(const Layer& layer, CUdeviceptr query, CUdeviceptr key, CUdeviceptr value,
                   CUdeviceptr out, int position, unsigned extent, CUdeviceptr position_slot) {
        launch(kernel.norm_rope, 8, 1, 256, query, layer.qnorm.data, 512, position, base, eps, position_slot);
        launch(kernel.norm_rope, 2, 1, 256, key, layer.knorm.data, 256, position, base, eps, position_slot);
        launch(kernel.cache_kv, 2, 1, 256, key, value, layer.keys, layer.values, position, position_slot);
        launch(kernel.scores, extent, 8, 256, query, layer.keys, scores, position + 1, capacity, position_slot);
        launch(kernel.read_attention, 8, 1, 256, query, layer.values, scores, out,
               position + 1, capacity, position_slot);
    }

    void recurrent(const Layer& layer, CUdeviceptr query, CUdeviceptr z_values,
                   CUdeviceptr a_values, CUdeviceptr b_values, CUdeviceptr out) {
        launch(kernel.convolve, 24, 1, 256, query, layer.history, layer.convolve.data);
        launch(kernel.delta, 16, 1, 128, query, z_values, a_values, b_values, layer.a.data,
               layer.bias.data, layer.rnorm.data, layer.state, out, eps);
    }

    void project_chunk(const Matrix& matrix, CUdeviceptr in, CUdeviceptr out, int tokens) {
        CUdeviceptr weights = matrix.data;
        if (matrix.type != 0) {
            if (matrix.cols * matrix.rows > max_layer_matrix)
                throw std::logic_error("prefill projection exceeds unpack scratch");
            weights = prefill_buffers.weights;
            launch(kernel.unpack_matrix, (matrix.cols + 255) / 256, matrix.rows, 256,
                   matrix.data, weights, matrix.cols, matrix.bytes, matrix.type);
        }
        // Contiguous token vectors are the columns of B and C in cuBLAS.
        const float one = 1, zero = 0;
        SPARK_CUDA(cublasSgemm(blas.value, CUBLAS_OP_T, CUBLAS_OP_N, matrix.rows, tokens,
                               matrix.cols, &one, reinterpret_cast<const float*>(weights),
                               matrix.cols, reinterpret_cast<const float*>(in), matrix.cols,
                               &zero, reinterpret_cast<float*>(out), matrix.rows));
    }

    void prefill_chunk(std::span<const TokenId> tokens) {
        const auto& b = prefill_buffers;
        const int count = static_cast<int>(tokens.size());
        const auto row = [](CUdeviceptr data, int index, int channels) {
            return data + static_cast<std::size_t>(index) * channels * sizeof(float);
        };
        for (int i = 0; i < count; ++i) gather(tokens[i], row(b.x, i, hidden));
        for (const auto& l : layers) {
            launch(kernel.rms, 1, count, 256, b.x, l.norm.data, b.n, hidden, eps);
            project_chunk(l.q, b.n, b.q, count);
            if (l.state) {
                project_chunk(l.z, b.n, b.z, count);
                project_chunk(l.alpha, b.n, b.alpha, count);
                project_chunk(l.beta, b.n, b.beta, count);
                // These updates are causal. The stream orders them before the output projection.
                for (int i = 0; i < count; ++i)
                    recurrent(l, row(b.q, i, l.q.rows), row(b.z, i, 2048), row(b.alpha, i, 16),
                              row(b.beta, i, 16), row(b.y, i, 2048));
            } else {
                project_chunk(l.k, b.n, b.k, count);
                project_chunk(l.v, b.n, b.v, count);
                for (int i = 0; i < count; ++i)
                    attention(l, row(b.q, i, l.q.rows), row(b.k, i, 512), row(b.v, i, 512),
                              row(b.y, i, 2048), length + i, length + i + 1, 0);
            }
            project_chunk(l.out, b.y, b.p, count);
            launch(kernel.residual, 4, count, 256, b.x, b.p, hidden);
            launch(kernel.rms, 1, count, 256, b.x, l.post.data, b.n, hidden, eps);
            project_chunk(l.gate, b.n, b.gate, count);
            project_chunk(l.up, b.n, b.up, count);
            launch(kernel.swiglu, 14, count, 256, b.gate, b.up, width);
            project_chunk(l.down, b.gate, b.p, count);
            launch(kernel.residual, 4, count, 256, b.x, b.p, hidden);
        }
        length += count;
    }

    std::span<const float> prefill(std::span<const TokenId> tokens) {
        if (tokens.empty() || tokens.size() > static_cast<std::size_t>(capacity - length) ||
            std::ranges::any_of(tokens, [](TokenId token) { return token >= vocab; }))
            throw std::out_of_range("invalid CUDA prompt tokens or context length");
        auto prefix = tokens.first(tokens.size() - 1);
        if (prefill_buffers.x) {
            while (!prefix.empty()) {
                const auto count = std::min(prefix.size(), static_cast<std::size_t>(prefill_chunk_size));
                prefill_chunk(prefix.first(count));
                prefix = prefix.subspan(count);
            }
        } else {
            for (const auto token : prefix) step(token, false);
        }
        return step(tokens.back(), true);
    }

    void forward(bool emit, bool trace, unsigned extent, CUdeviceptr position_slot) {
        const int position = position_slot ? 0 : length;
        if (trace) profile.mark(0, stream.value);
        for (std::size_t index = 0; index < layers.size(); ++index) {
            const auto& l = layers[index];
            launch(kernel.rms, 1, 1, 256, x, l.norm.data, n, hidden, eps);
            mv(l.q, n, q);
            if (l.state) {
                mv(l.z, n, z);
                mv(l.alpha, n, alpha);
                mv(l.beta, n, beta);
                recurrent(l, q, z, alpha, beta, y);
            } else {
                mv(l.k, n, k);
                mv(l.v, n, v);
                attention(l, q, k, v, y, position, extent, position_slot);
            }
            mv(l.out, y, p);
            launch(kernel.residual, 4, 1, 256, x, p, hidden);
            if (trace) profile.mark(2 * index + 1, stream.value);
            launch(kernel.rms, 1, 1, 256, x, l.post.data, n, hidden, eps);
            mv(l.gate, n, gate);
            mv(l.up, n, up);
            launch(kernel.swiglu, 14, 1, 256, gate, up, width);
            mv(l.down, gate, p);
            launch(kernel.residual, 4, 1, 256, x, p, hidden);
            if (trace) profile.mark(2 * index + 2, stream.value);
        }
        if (!emit) return;
        launch(kernel.rms, 1, 1, 256, x, last_norm.data, n, hidden, eps);
        mv(embedding, n, logits);
        if (trace) profile.mark(49, stream.value);
    }

    void replay_graph() {
        const auto count = static_cast<unsigned>(length + 1);
        const auto bucket = std::bit_ceil(count);
        const auto index = graph_mode == DecodeGraph::single ? 13U : std::bit_width(bucket) - 1U;
        const auto extent = graph_mode == DecodeGraph::single ? static_cast<unsigned>(capacity)
                                                             : bucket;
        auto& exec = graphs.entries[index];
        if (!exec) {
            CUgraph graph = nullptr;
            SPARK_CUDA(cuStreamBeginCapture(stream.value, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
            try {
                forward(true, false, extent, cursor);
                SPARK_CUDA(cuStreamEndCapture(stream.value, &graph));
            } catch (...) {
                if (!graph) SPARK_CUDA_CLEANUP(cuStreamEndCapture(stream.value, &graph));
                if (graph) SPARK_CUDA_CLEANUP(cuGraphDestroy(graph));
                throw;
            }
            try {
                SPARK_CUDA(cuGraphInstantiate(&exec, graph, 0));
            } catch (...) {
                SPARK_CUDA_CLEANUP(cuGraphDestroy(graph));
                throw;
            }
            SPARK_CUDA(cuGraphDestroy(graph));
        }
        std::memcpy(host_cursor.value, &length, sizeof(length));
        SPARK_CUDA(cuMemcpyHtoDAsync(cursor, host_cursor.value, sizeof(length), stream.value));
        SPARK_CUDA(cuGraphLaunch(exec, stream.value));
    }

    std::span<const float> step(TokenId token, bool emit) {
        if (token >= vocab || length >= capacity) throw std::out_of_range("invalid CUDA token or position");
        const bool trace = emit && emitted == 1 && profile.enabled();
        gather(token, x);
        if (emit && emitted > 0 && graph_mode != DecodeGraph::off) replay_graph();
        else forward(emit, trace, length + 1, 0);
        ++length;
        if (!emit) return {};
        SPARK_CUDA(cuMemcpyDtoHAsync(host.value, logits, host.size * sizeof(float), stream.value));
        SPARK_CUDA(cuStreamSynchronize(stream.value));
        ++emitted;
        if (trace) profile.report();
        return {host.value, host.size};
    }
};

GpuModel::GpuModel(const Gguf& gguf, const std::filesystem::path& path, std::size_t context,
                   bool profile, DecodeGraph graph, bool packed, bool batch_prefill)
    : impl_(std::make_unique<Impl>(gguf, path, context, profile, graph, packed, batch_prefill)) {}
GpuModel::~GpuModel() = default;
void GpuModel::reset() { impl_->reset(); }
std::span<const float> GpuModel::prefill(std::span<const TokenId> tokens) { return impl_->prefill(tokens); }
std::span<const float> GpuModel::step(TokenId token, bool logits) { return impl_->step(token, logits); }

void test_qwen_cuda_ops() {
    Context context;
    Stream stream;
    Blas blas(stream.value);
    Module module;
    Kernels kernel(module);
    DeviceArena arena(4 * 1024 * 1024);
    std::mt19937 rng(4401);
    std::uniform_real_distribution<float> uniform(-0.5F, 0.5F);
    const auto data = [&](std::size_t size) {
        std::vector<float> values(size);
        std::ranges::generate(values, [&] { return uniform(rng); });
        return values;
    };
    const auto put = [&](std::span<const float> values) {
        const auto dst = arena.allocate(values.size_bytes());
        SPARK_CUDA(cuMemcpyHtoD(dst, values.data(), values.size_bytes()));
        return dst;
    };
    const auto run = [&](CUfunction fn, unsigned gx, unsigned gy, unsigned block, auto... args) {
        void* params[] = {static_cast<void*>(&args)...};
        SPARK_CUDA(cuLaunchKernel(fn, gx, gy, 1, block, 1, 1, 0, stream.value, params, nullptr));
    };
    const auto expect = [&](const char* label, CUdeviceptr src, std::span<const float> expected,
                            float tolerance = 1e-3F) {
        SPARK_CUDA(cuStreamSynchronize(stream.value));
        std::vector<float> actual(expected.size());
        SPARK_CUDA(cuMemcpyDtoH(actual.data(), src, expected.size_bytes()));
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) >= tolerance)
                throw std::runtime_error(std::string(label) + " failed at element " + std::to_string(i));
        }
    };
    const auto dot = [](std::span<const float> a, std::span<const float> b) {
        return std::inner_product(a.begin(), a.end(), b.begin(), 0.0F);
    };
    const auto sigmoid = [](float x) { return 1.0F / (1.0F + std::exp(-x)); };
    constexpr float eps = 1e-6F;
    {
        const auto x = data(1024), w = data(1024);
        auto expected = x;
        const float scale = 1.0F / std::sqrt(dot(x, x) / 1024 + eps);
        for (std::size_t i = 0; i < x.size(); ++i) expected[i] *= scale * w[i];
        const auto output = arena.allocate(x.size() * sizeof(float));
        run(kernel.rms, 1, 1, 256, put(x), put(w), output, 1024, eps);
        expect("rms", output, expected);
        for (std::size_t i = 0; i < x.size(); ++i) expected[i] += x[i];
        run(kernel.residual, 4, 1, 256, output, put(x), 1024);
        expect("residual", output, expected);
        for (std::size_t i = 0; i < x.size(); ++i) expected[i] *= sigmoid(expected[i]) * w[i];
        run(kernel.swiglu, 4, 1, 256, output, put(w), 1024);
        expect("swiglu", output, expected);
    }
    {
        constexpr int columns = 4096, rows = 5;
        const auto x = data(columns), w = data(columns * rows);
        const auto gpu_x = put(x), gpu_w = put(w);
        const auto output = arena.allocate(rows * sizeof(float));
        const float one = 1, zero = 0;
        SPARK_CUDA(cublasSgemv(blas.value, CUBLAS_OP_T, columns, rows, &one,
            reinterpret_cast<const float*>(gpu_w), columns, reinterpret_cast<const float*>(gpu_x),
            1, &zero, reinterpret_cast<float*>(output), 1));
        std::array<float, rows> expected{};
        for (int r = 0; r < rows; ++r)
            expected[r] = dot(x, std::span(w).subspan(r * columns, columns));
        expect("cublas projection", output, expected, 1e-2F);
    }
    {
        constexpr int columns = 1024, rows = 5;
        const auto x = data(columns);
        const auto x_gpu = put(x);
        for (const int type : {8, 12, 13, 14}) {
            const auto layout = gguf_encoding(type);
            const auto row_bytes = columns / layout.elements * layout.bytes;
            std::vector<std::uint8_t> packed(rows * row_bytes);
            for (auto& byte : packed) byte = static_cast<std::uint8_t>(rng() % 256);
            for (std::size_t start = 0; start < packed.size(); start += layout.bytes) {
                const auto scale = start + (type == 14 ? 208 : 0);
                packed[scale] = 0;
                packed[scale + 1] = 0x24;
                if (type == 12 || type == 13) {
                    packed[start + 2] = 0;
                    packed[start + 3] = 0x20;
                }
            }
            const auto weights_gpu = arena.allocate(packed.size());
            SPARK_CUDA(cuMemcpyHtoD(weights_gpu, packed.data(), packed.size()));
            const auto output = arena.allocate(rows * sizeof(float));
            run(kernel.packed_matvec, 2, 1, 128, weights_gpu, x_gpu, output, columns, rows,
                static_cast<int>(row_bytes), type);
            std::array<float, rows> expected{};
            std::array<float, columns> row{};
            for (std::size_t r = 0; r < rows; ++r) {
                unpack_gguf_row(type, std::span(packed).subspan(r * row_bytes, row_bytes), row);
                expected[r] = dot(x, row);
            }
            expect("packed GPU projection", output, expected, 1e-2F);
            const auto embedding = arena.allocate(columns * sizeof(float));
            run(kernel.packed_embedding, 4, 1, 256, weights_gpu, embedding, type);
            unpack_gguf_row(type, std::span(packed).first(row_bytes), row);
            expect("packed GPU embedding", embedding, row);
        }
    }
    {
        const auto x = data(6144), history = data(3 * 6144), filters = data(4 * 6144);
        auto expected = x, next = history;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const auto h = std::span(history).subspan(3 * i, 3);
            const auto f = std::span(filters).subspan(4 * i, 4);
            const float value = dot(h, f.first(3)) + x[i] * f[3];
            expected[i] = value * sigmoid(value);
            next[3 * i] = h[1]; next[3 * i + 1] = h[2]; next[3 * i + 2] = x[i];
        }
        const auto x_gpu = put(x), history_gpu = put(history);
        run(kernel.convolve, 24, 1, 256, x_gpu, history_gpu, put(filters));
        expect("causal convolution", x_gpu, expected);
        expect("convolution history", history_gpu, next);
    }
    {
        constexpr int position = 7;
        const auto input = data(512), w = data(256), v = data(512), q = data(4096);
        auto k = input;
        for (int h = 0; h < 2; ++h) {
            auto head = std::span(k).subspan(h * 256, 256);
            const auto scale = 1.0F / std::sqrt(dot(head, head) / 256 + eps);
            for (unsigned i = 0; i < head.size(); ++i) head[i] *= w[i] * scale;
            for (int i = 0; i < 32; ++i) {
                const float angle = position / std::pow(1e7F, i / 32.0F);
                const float a = head[i], b = head[i + 32];
                head[i] = a * std::cos(angle) - b * std::sin(angle);
                head[i + 32] = b * std::cos(angle) + a * std::sin(angle);
            }
        }
        const auto kg = put(input);
        run(kernel.norm_rope, 2, 1, 256, kg, put(w), 256, position, 1e7F, eps, CUdeviceptr{0});
        expect("normalization and RoPE", kg, k);
        auto keys = data(2 * 512), values = data(2 * 512);
        const auto key_cache = put(keys), value_cache = put(values), qg = put(q);
        run(kernel.cache_kv, 2, 1, 256, kg, put(v), key_cache, value_cache, 1, CUdeviceptr{0});
        std::ranges::copy(k, keys.begin() + 512);
        std::ranges::copy(v, values.begin() + 512);
        expect("key cache", key_cache, keys);
        expect("value cache", value_cache, values);
        const auto score_gpu = arena.allocate(16 * sizeof(float));
        run(kernel.scores, 2, 8, 256, qg, key_cache, score_gpu, 2, 2, CUdeviceptr{0});
        std::array<float, 16> scores{};
        std::array<float, 2048> attention{};
        for (int h = 0; h < 8; ++h) {
            const auto query = std::span(q).subspan(h * 512, 256);
            const auto group = (h / 4) * 256;
            const auto s0 = dot(query, std::span(keys).subspan(group, 256)) / 16;
            const auto s1 = dot(query, std::span(keys).subspan(512 + group, 256)) / 16;
            scores[h * 2] = s0;
            scores[h * 2 + 1] = s1;
            const auto a = sigmoid(s0 - s1);
            for (int i = 0; i < 256; ++i)
                attention[h * 256 + i] = (a * values[group + i] + (1 - a) * values[512 + group + i])
                                         * sigmoid(q[h * 512 + 256 + i]);
        }
        expect("decode attention scores", score_gpu, scores);
        const auto output = arena.allocate(attention.size() * sizeof(float));
        run(kernel.read_attention, 8, 1, 256, qg, value_cache, score_gpu, output, 2, 2, CUdeviceptr{0});
        expect("decode attention readout", output, attention);
    }
    {
        const auto qkv = data(6144), z = data(2048), alpha = data(16), beta = data(16);
        const auto bias = data(16), w = data(128);
        std::vector<float> a(16, -0.7F), state(16 * 128 * 128), out(2048);
        std::array<float, 128> scratch{};
        const auto state_gpu = put(state), out_gpu = put(out);
        const auto qkv_gpu = put(qkv), z_gpu = put(z), alpha_gpu = put(alpha), beta_gpu = put(beta);
        const auto a_gpu = put(a), bias_gpu = put(bias), w_gpu = put(w);
        for (int t = 0; t < 2; ++t) {
            run(kernel.delta, 16, 1, 128, qkv_gpu, z_gpu, alpha_gpu, beta_gpu, a_gpu, bias_gpu,
                w_gpu, state_gpu, out_gpu, eps);
            for (int h = 0; h < 16; ++h) {
                std::array<float, 128> query{}, key{};
                std::ranges::copy(std::span(qkv).subspan(h * 128, 128), query.begin());
                std::ranges::copy(std::span(qkv).subspan(2048 + h * 128, 128), key.begin());
                const auto q_scale = 1 / std::sqrt(dot(query, query) + eps);
                const auto k_scale = 1 / std::sqrt(dot(key, key) + eps);
                for (unsigned i = 0; i < 128; ++i) { query[i] *= q_scale; key[i] *= k_scale; }
                auto head = std::span(out).subspan(h * 128, 128);
                const auto decay = std::exp(a[h] * std::log1p(std::exp(alpha[h] + bias[h])));
                delta_net_step(std::span(state).subspan(h * 128 * 128, 128 * 128),
                    query, key, std::span(qkv).subspan(4096 + h * 128, 128),
                    decay, sigmoid(beta[h]), head, scratch);
                const auto scale = 1 / std::sqrt(dot(head, head) / 128 + eps);
                for (unsigned i = 0; i < 128; ++i)
                    head[i] *= scale * w[i] * z[h * 128 + i] * sigmoid(z[h * 128 + i]);
            }
            expect("Gated DeltaNet readout", out_gpu, out);
            expect("Gated DeltaNet state", state_gpu, state);
        }
    }
}

}  // namespace spark
