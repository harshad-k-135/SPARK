#include "engine/transformer/cpu_model.hpp"
#include "engine/model_loader/packed_weights.hpp"
#include "engine/transformer/delta_net.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace spark {
namespace {

constexpr std::size_t hidden = 1024, inner = 3584, dh = 256, qh = 8, kvh = 2;
constexpr std::size_t sh = 16, sd = 128, sw = sh * sd, conv = 4;
using Vec = std::vector<float>;
using View = std::span<const float>;
using Mut = std::span<float>;

std::size_t checked_context(std::size_t context) {
    if (context == 0 || context > 4096) throw std::invalid_argument("invalid context size");
    return context;
}

float dot(View a, View b) {
    float sum[8]{};
    std::size_t x = 0;
    for (; x + 8 <= a.size(); x += 8)
        for (std::size_t j = 0; j < 8; ++j) sum[j] += a[x + j] * b[x + j];
    float total = (sum[0] + sum[1] + sum[2] + sum[3]) + (sum[4] + sum[5] + sum[6] + sum[7]);
    for (; x < a.size(); ++x) total += a[x] * b[x];
    return total;
}

void rms(View in, View weight, Mut out, float eps) {
    const auto scale = 1.0F / std::sqrt(dot(in, in) / static_cast<float>(in.size()) + eps);
    for (std::size_t i = 0; i < in.size(); ++i) out[i] = in[i] * scale * weight[i];
}

void l2(Mut x, float eps) {
    const auto scale = 1.0F / std::sqrt(dot(x, x) + eps);
    for (auto& item : x) item *= scale;
}

float sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }
float silu(float x) { return x * sigmoid(x); }
float softplus(float x) { return std::max(0.0F, x) + std::log1p(std::exp(-std::abs(x))); }

struct Matrix {
    std::size_t columns = 0, rows = 0;
    Vec data;

    void mv(View x, Mut y) const {
#ifdef _OPENMP
        #pragma omp parallel for if(rows * columns >= (1U << 19)) schedule(static)
#endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(rows); ++i)
            y[i] = dot(View(data).subspan(static_cast<std::size_t>(i) * columns, columns), x);
    }

    View row(std::size_t i) const { return View(data).subspan(i * columns, columns); }
};

Matrix load_matrix(std::ifstream& file, const Gguf& gguf, const std::string& name,
                   std::size_t columns, std::size_t rows = 1) {
    const auto& t = gguf.tensors().at(name);
    if (t.shape[0] != columns || t.shape[1] != rows || t.rank > 2)
        throw std::runtime_error("unsupported tensor shape for " + name);
    const auto row_bytes = static_cast<std::size_t>(t.bytes / rows);
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(t.bytes));
    file.seekg(static_cast<std::streamoff>(t.file_offset));
    if (!file.read(reinterpret_cast<char*>(packed.data()), static_cast<std::streamsize>(packed.size())))
        throw std::runtime_error("could not load " + name);
    Matrix result{columns, rows, Vec(columns * rows)};
#ifdef _OPENMP
    #pragma omp parallel for if(rows * columns >= (1U << 19)) schedule(static)
#endif
    for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
        const auto row = static_cast<std::size_t>(r);
        unpack_gguf_row(t.type, std::span(packed).subspan(row * row_bytes, row_bytes),
                        Mut(result.data).subspan(row * columns, columns));
    }
    return result;
}

struct Layer {
    Matrix norm, post, up, gate, down, q, k, v, out, qnorm, knorm;
    Matrix convolve, z, alpha, beta, decay, bias, rnorm;
    Vec state, history, keys, values;

    bool recurrent() const { return !state.empty(); }
};

void rotate(Mut x, const std::array<float, 32>& c, const std::array<float, 32>& s) {
    for (std::size_t i = 0; i < c.size(); ++i) {
        const auto a = x[i], b = x[i + c.size()];
        x[i] = c[i] * a - s[i] * b;
        x[i + c.size()] = c[i] * b + s[i] * a;
    }
}

}  // namespace

struct CpuModel::Impl {
    std::size_t context, position = 0;
    float eps;
    Matrix embedding, final_norm;
    std::vector<Layer> layers;
    Vec x = Vec(hidden), norm = Vec(hidden), projected = Vec(hidden);
    Vec q = Vec(3 * sw), k = Vec(kvh * dh), v = Vec(kvh * dh), y = Vec(sw);
    Vec gate = Vec(inner), up = Vec(inner), z = Vec(sw), scores, logits;
    std::array<float, sh> alpha{}, beta{};
    std::array<float, sd> scratch{};
    std::array<float, 32> cosines{}, sines{};

    Impl(const Gguf& gguf, const std::filesystem::path& path, std::size_t capacity)
        : context(checked_context(capacity)), eps(static_cast<float>(std::get<double>(
          gguf.metadata("qwen35.attention.layer_norm_rms_epsilon")))), scores(context) {
        const auto check = [&](std::string_view name, std::uint64_t expected) {
            if (std::get<std::uint64_t>(gguf.metadata(name)) != expected)
                throw std::runtime_error("unsupported Qwen3.5 metadata " + std::string(name));
        };
        if (std::get<std::string>(gguf.metadata("general.architecture")) != "qwen35")
            throw std::runtime_error("expected Qwen3.5 GGUF");
        check("qwen35.embedding_length", hidden);
        check("qwen35.feed_forward_length", inner);
        check("qwen35.attention.head_count", qh);
        check("qwen35.attention.head_count_kv", kvh);
        check("qwen35.attention.key_length", dh);
        check("qwen35.attention.value_length", dh);
        check("qwen35.rope.dimension_count", 64);
        check("qwen35.ssm.group_count", sh);
        check("qwen35.ssm.time_step_rank", sh);
        check("qwen35.ssm.state_size", sd);
        check("qwen35.ssm.inner_size", sw);
        check("qwen35.ssm.conv_kernel", conv);
        check("qwen35.block_count", 25);
        check("qwen35.nextn_predict_layers", 1);
        check("qwen35.full_attention_interval", 4);
        if (gguf.tensors().contains("output.weight"))
            throw std::runtime_error("expected tied Qwen3.5 output weights");
        std::ifstream file(path, std::ios::binary);
        const auto vocab = std::get<GgufArray>(gguf.metadata("tokenizer.ggml.tokens")).count;
        embedding = load_matrix(file, gguf, "token_embd.weight", hidden, vocab);
        final_norm = load_matrix(file, gguf, "output_norm.weight", hidden);
        logits.resize(vocab);

        for (std::size_t i = 0; i < 24; ++i) {
            const auto prefix = "blk." + std::to_string(i) + ".";
            const auto m = [&](const char* suffix, std::size_t cols, std::size_t rows = 1) {
                return load_matrix(file, gguf, prefix + suffix, cols, rows);
            };
            auto& l = layers.emplace_back();
            l.norm = m("attn_norm.weight", hidden);
            l.post = m("post_attention_norm.weight", hidden);
            l.up = m("ffn_up.weight", hidden, inner);
            l.gate = m("ffn_gate.weight", hidden, inner);
            l.down = m("ffn_down.weight", inner, hidden);
            if ((i + 1) % 4 == 0) {
                l.q = m("attn_q.weight", hidden, 2 * qh * dh);
                l.k = m("attn_k.weight", hidden, kvh * dh);
                l.v = m("attn_v.weight", hidden, kvh * dh);
                l.out = m("attn_output.weight", qh * dh, hidden);
                l.qnorm = m("attn_q_norm.weight", dh);
                l.knorm = m("attn_k_norm.weight", dh);
                l.keys.resize(context * kvh * dh);
                l.values.resize(l.keys.size());
            } else {
                l.q = m("attn_qkv.weight", hidden, 3 * sw);
                l.z = m("attn_gate.weight", hidden, sw);
                l.alpha = m("ssm_alpha.weight", hidden, sh);
                l.beta = m("ssm_beta.weight", hidden, sh);
                l.decay = m("ssm_a", sh);
                l.bias = m("ssm_dt.bias", sh);
                l.convolve = m("ssm_conv1d.weight", conv, 3 * sw);
                l.rnorm = m("ssm_norm.weight", sd);
                l.out = m("ssm_out.weight", sw, hidden);
                l.state.resize(sh * sd * sd);
                l.history.resize(3 * sw * (conv - 1));
            }
        }
        rope_base = static_cast<float>(std::get<double>(gguf.metadata("qwen35.rope.freq_base")));
    }

    float rope_base = 0;

    void recurrent(Layer& l) {
        l.q.mv(norm, q);
        l.z.mv(norm, z);
        l.alpha.mv(norm, alpha);
        l.beta.mv(norm, beta);
        for (std::size_t i = 0; i < q.size(); ++i) {
            auto hist = Mut(l.history).subspan(i * (conv - 1), conv - 1);
            const auto filter = l.convolve.row(i);
            const float total = dot(hist, filter.first(conv - 1)) + q[i] * filter.back();
            std::move(hist.begin() + 1, hist.end(), hist.begin());
            hist.back() = q[i];
            q[i] = silu(total);
        }
        for (std::size_t h = 0; h < sh; ++h) {
            const auto query = Mut(q).subspan(h * sd, sd);
            const auto key = Mut(q).subspan(sw + h * sd, sd);
            const auto value = View(q).subspan(2 * sw + h * sd, sd);
            auto out = Mut(y).subspan(h * sd, sd);
            l2(query, eps);
            l2(key, eps);
            const auto decay = std::exp(l.decay.data[h] * softplus(alpha[h] + l.bias.data[h]));
            delta_net_step(Mut(l.state).subspan(h * sd * sd, sd * sd),
                           query, key, value, decay, sigmoid(beta[h]), out, scratch);
            rms(out, l.rnorm.data, out, eps);
            for (std::size_t j = 0; j < sd; ++j) out[j] *= silu(z[h * sd + j]);
        }
    }

    void attention(Layer& l) {
        l.q.mv(norm, q);
        l.k.mv(norm, k);
        l.v.mv(norm, v);
        for (std::size_t h = 0; h < kvh; ++h) {
            auto key = Mut(k).subspan(h * dh, dh);
            rms(key, l.knorm.data, key, eps);
            rotate(key, cosines, sines);
        }
        std::ranges::copy(k, l.keys.begin() + position * kvh * dh);
        std::ranges::copy(v, l.values.begin() + position * kvh * dh);
        auto used = Mut(scores).first(position + 1);
        for (std::size_t h = 0; h < qh; ++h) {
            auto query = Mut(q).subspan(2 * h * dh, dh);
            const auto gate_values = View(q).subspan((2 * h + 1) * dh, dh);
            rms(query, l.qnorm.data, query, eps);
            rotate(query, cosines, sines);
            const auto group = h / (qh / kvh);
            for (std::size_t t = 0; t <= position; ++t)
                used[t] = dot(query, View(l.keys).subspan((t * kvh + group) * dh, dh)) / 16.0F;
            const auto max = std::ranges::max(used);
            float mass = 0;
            for (auto& score : used) {
                score = std::exp(score - max);
                mass += score;
            }
            auto out = Mut(y).subspan(h * dh, dh);
            std::ranges::fill(out, 0.0F);
            for (std::size_t t = 0; t <= position; ++t) {
                const auto row = View(l.values).subspan((t * kvh + group) * dh, dh);
                const auto p = used[t] / mass;
                for (std::size_t j = 0; j < dh; ++j) out[j] += p * row[j];
            }
            for (std::size_t j = 0; j < dh; ++j) out[j] *= sigmoid(gate_values[j]);
        }
    }

    View step(TokenId token, bool emit) {
        if (token >= embedding.rows || position >= context)
            throw std::out_of_range("token or position outside Qwen session");
        std::ranges::copy(embedding.row(token), x.begin());
        for (std::size_t i = 0; i < cosines.size(); ++i) {
            const auto angle = static_cast<float>(position) /
                               std::pow(rope_base, static_cast<float>(i) / 32.0F);
            cosines[i] = std::cos(angle);
            sines[i] = std::sin(angle);
        }
        for (auto& l : layers) {
            rms(x, l.norm.data, norm, eps);
            if (l.recurrent()) recurrent(l); else attention(l);
            l.out.mv(y, projected);
            for (std::size_t j = 0; j < hidden; ++j) x[j] += projected[j];
            rms(x, l.post.data, norm, eps);
            l.gate.mv(norm, gate);
            l.up.mv(norm, up);
            for (std::size_t j = 0; j < inner; ++j) gate[j] = silu(gate[j]) * up[j];
            l.down.mv(gate, projected);
            for (std::size_t j = 0; j < hidden; ++j) x[j] += projected[j];
        }
        ++position;
        if (!emit) return {};
        rms(x, final_norm.data, norm, eps);
        embedding.mv(norm, logits);
        return logits;
    }
};

CpuModel::CpuModel(const Gguf& gguf, const std::filesystem::path& path, std::size_t context)
    : impl_(std::make_unique<Impl>(gguf, path, context)) {}
CpuModel::~CpuModel() = default;

void CpuModel::reset() {
    impl_->position = 0;
    for (auto& layer : impl_->layers) {
        std::ranges::fill(layer.state, 0.0F);
        std::ranges::fill(layer.history, 0.0F);
    }
}

std::span<const float> CpuModel::step(TokenId token, bool logits) { return impl_->step(token, logits); }

}  // namespace spark
