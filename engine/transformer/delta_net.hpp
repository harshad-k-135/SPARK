#pragma once

#include <cstddef>
#include <span>

namespace spark {

// One normalized query/key/value head. State is [key_width][value_width]
// with contiguous value rows. Decay is supplied as the already exponentiated gate.
void delta_net_step(std::span<float> state, std::span<const float> query,
                    std::span<const float> key, std::span<const float> value,
                    float decay, float beta, std::span<float> output,
                    std::span<float> delta_scratch);

}  // namespace spark
