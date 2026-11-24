#include "engine/transformer/delta_net.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spark {

void delta_net_step(std::span<float> state, std::span<const float> query,
                    std::span<const float> key, std::span<const float> value,
                    float decay, float beta, std::span<float> output,
                    std::span<float> delta_scratch) {
    const auto key_width = key.size();
    const auto value_width = value.size();
    if (key_width == 0 || value_width == 0 || query.size() != key_width ||
        output.size() != value_width || delta_scratch.size() != value_width ||
        state.size() % value_width != 0 || state.size() / value_width != key_width) {
        throw std::invalid_argument("Gated DeltaNet head dimensions disagree");
    }

    std::ranges::fill(delta_scratch, 0.0F);
    for (std::size_t i = 0; i < key_width; ++i) {
        auto row = state.subspan(i * value_width, value_width);
        for (std::size_t j = 0; j < value_width; ++j) {
            row[j] *= decay;
            delta_scratch[j] += key[i] * row[j];
        }
    }
    for (std::size_t j = 0; j < value_width; ++j) {
        delta_scratch[j] = (value[j] - delta_scratch[j]) * beta;
    }

    std::ranges::fill(output, 0.0F);
    const auto inverse_sqrt_width = 1.0F / std::sqrt(static_cast<float>(key_width));
    for (std::size_t i = 0; i < key_width; ++i) {
        auto row = state.subspan(i * value_width, value_width);
        const auto scaled_query = query[i] * inverse_sqrt_width;
        for (std::size_t j = 0; j < value_width; ++j) {
            row[j] += key[i] * delta_scratch[j];
            output[j] += scaled_query * row[j];
        }
    }
}

}  // namespace spark
