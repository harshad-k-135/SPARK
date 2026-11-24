#include "engine/transformer/cpu_forward.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace spark {
namespace {

std::size_t cache_size(const ModelConfig& config, std::size_t context) {
    if (context == 0 || context > 4096 || context > config.max_positions) {
        throw std::invalid_argument("context exceeds the model or SPARK context limit");
    }
    const auto per_layer = context * config.kv_width();
    if (config.layer_count > std::numeric_limits<std::size_t>::max() / per_layer) {
        throw std::invalid_argument("KV cache dimensions overflow");
    }
    return config.layer_count * per_layer;
}

float dot(std::span<const float> left, std::span<const float> right) {
    float partial[8]{};
    std::size_t i = 0;
    for (; i + 8 <= left.size(); i += 8) {
        for (std::size_t j = 0; j < 8; ++j) {
            partial[j] += left[i + j] * right[i + j];
        }
    }
    float result = ((partial[0] + partial[1]) + (partial[2] + partial[3])) +
                   ((partial[4] + partial[5]) + (partial[6] + partial[7]));
    for (; i < left.size(); ++i) {
        result += left[i] * right[i];
    }
    return result;
}

void project(Weights weights, std::span<const float> input, std::span<float> output) {
#ifdef _OPENMP
    #pragma omp parallel for if(weights.size() >= (1U << 19)) schedule(static)
#endif
    for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(output.size()); ++row) {
        const auto offset = static_cast<std::size_t>(row) * input.size();
        output[row] = dot(weights.subspan(offset, input.size()), input);
    }
}

void rmsnorm(std::span<const float> input, Weights weight, float epsilon,
             std::span<float> output) {
    const auto mean_square = dot(input, input) / static_cast<float>(input.size());
    const auto inverse_rms = 1.0F / std::sqrt(mean_square + epsilon);
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = (input[i] * inverse_rms) * weight[i];
    }
}

float scaled_frequency(const ModelConfig& config, std::size_t pair) {
    const auto exponent = 2.0F * static_cast<float>(pair) / static_cast<float>(config.head_dim);
    const auto freq = 1.0F / std::pow(config.rope_theta, exponent);
    const auto wavelength = 2.0F * std::numbers::pi_v<float> / freq;
    const auto high_boundary = config.rope_original_context / config.rope_high_freq;
    const auto low_boundary = config.rope_original_context / config.rope_low_freq;
    if (wavelength < high_boundary) {
        return freq;
    }
    if (wavelength > low_boundary) {
        return freq / config.rope_factor;
    }
    const auto mix = (config.rope_original_context / wavelength - config.rope_low_freq) /
                     (config.rope_high_freq - config.rope_low_freq);
    return (1.0F - mix) * (freq / config.rope_factor) + mix * freq;
}

void rotate(std::span<float> activations, std::span<const float> cosine,
            std::span<const float> sine) {
    const auto half = cosine.size();
    const auto dim = 2 * half;
    for (std::size_t head = 0; head < activations.size(); head += dim) {
        for (std::size_t i = 0; i < half; ++i) {
            const auto low = activations[head + i];
            const auto high = activations[head + half + i];
            activations[head + i] = low * cosine[i] - high * sine[i];
            activations[head + half + i] = high * cosine[i] + low * sine[i];
        }
    }
}

void add(std::span<float> destination, std::span<const float> residual) {
    for (std::size_t i = 0; i < destination.size(); ++i) {
        destination[i] += residual[i];
    }
}

}  // namespace

CpuForward::CpuForward(const Model& model, std::size_t context)
    : model_(model), context_(context),
      keys_(cache_size(model.config(), context)), values_(keys_.size()),
      cosine_(context * (model.config().head_dim / 2)), sine_(cosine_.size()),
      hidden_(model.config().hidden), normalized_(hidden_.size()), query_(hidden_.size()),
      key_(model.config().kv_width()), value_(key_.size()), attended_(hidden_.size()),
      projected_(hidden_.size()), gate_(model.config().intermediate), up_(gate_.size()),
      scores_(context), logits_(model.config().vocab) {
    const auto half = model_.config().head_dim / 2;
    for (std::size_t i = 0; i < half; ++i) {
        const auto frequency = scaled_frequency(model_.config(), i);
        for (std::size_t position = 0; position < context_; ++position) {
            const auto angle = static_cast<float>(position) * frequency;
            cosine_[position * half + i] = std::cos(angle);
            sine_[position * half + i] = std::sin(angle);
        }
    }
}

std::span<float> CpuForward::cache_row(std::span<float> cache, std::size_t layer,
                                      std::size_t head, std::size_t position) {
    const auto& config = model_.config();
    const auto offset = ((layer * config.kv_heads + head) * context_ + position) * config.head_dim;
    return cache.subspan(offset, config.head_dim);
}

void CpuForward::attention(std::size_t layer, std::size_t position) {
    const auto& config = model_.config();
    const auto dim = config.head_dim;
    const auto group = config.heads / config.kv_heads;
    const auto scale = 1.0F / std::sqrt(static_cast<float>(dim));
    const auto scores = std::span(scores_).first(position + 1);
    for (std::size_t head = 0; head < config.heads; ++head) {
        const auto query = std::span(query_).subspan(head * dim, dim);
        const auto shared_head = head / group;
        for (std::size_t t = 0; t <= position; ++t) {
            scores[t] = dot(query, cache_row(keys_, layer, shared_head, t)) * scale;
        }
        const auto max_score = std::ranges::max(scores);
        float denominator = 0;
        for (auto& score : scores) {
            score = std::exp(score - max_score);
            denominator += score;
        }
        const auto output = std::span(attended_).subspan(head * dim, dim);
        std::ranges::fill(output, 0.0F);
        for (std::size_t t = 0; t <= position; ++t) {
            const auto row = cache_row(values_, layer, shared_head, t);
            const auto probability = scores[t] / denominator;
            for (std::size_t i = 0; i < dim; ++i) {
                output[i] += probability * row[i];
            }
        }
    }
}

void CpuForward::push(TokenId token, std::size_t position) {
    const auto& config = model_.config();
    const auto dim = config.head_dim;
    const auto half = dim / 2;
    const auto cosine = std::span(cosine_).subspan(position * half, half);
    const auto sine = std::span(sine_).subspan(position * half, half);
    std::ranges::copy(model_.embedding().subspan(token * config.hidden, config.hidden),
                      hidden_.begin());

    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        const auto& weights = model_.layers()[layer];
        rmsnorm(hidden_, weights.attention_norm, config.rms_epsilon, normalized_);
        project(weights.query, normalized_, query_);
        project(weights.key, normalized_, key_);
        project(weights.value, normalized_, value_);
        rotate(query_, cosine, sine);
        rotate(key_, cosine, sine);
        for (std::size_t head = 0; head < config.kv_heads; ++head) {
            std::ranges::copy(std::span(key_).subspan(head * dim, dim),
                              cache_row(keys_, layer, head, position).begin());
            std::ranges::copy(std::span(value_).subspan(head * dim, dim),
                              cache_row(values_, layer, head, position).begin());
        }
        attention(layer, position);
        project(weights.attention_out, attended_, projected_);
        add(hidden_, projected_);

        rmsnorm(hidden_, weights.mlp_norm, config.rms_epsilon, normalized_);
        project(weights.gate, normalized_, gate_);
        project(weights.up, normalized_, up_);
        for (std::size_t i = 0; i < gate_.size(); ++i) {
            const auto silu = gate_[i] / (1.0F + std::exp(-gate_[i]));
            gate_[i] = silu * up_[i];
        }
        project(weights.down, gate_, projected_);
        add(hidden_, projected_);
    }
}

std::span<const float> CpuForward::logits() {
    rmsnorm(hidden_, model_.final_norm(), model_.config().rms_epsilon, normalized_);
    project(model_.lm_head(), normalized_, logits_);
    return logits_;
}

}  // namespace spark
