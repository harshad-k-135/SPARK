#include "engine/transformer/delta_net.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void expect(std::span<const float> values, std::span<const float> expected) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (std::abs(values[i] - expected[i]) >= 1e-6F) {
            throw std::runtime_error("Gated DeltaNet recurrence value differs from reference");
        }
    }
}

}  // namespace

int main() {
    std::array<float, 8> state{};
    std::array<float, 2> out{};
    std::array<float, 2> scratch{};
    spark::delta_net_step(state, std::array{1.0F, 0.0F, 0.0F, 0.0F},
                         std::array{1.0F, 0.0F, 0.0F, 0.0F}, std::array{4.0F, 8.0F},
                         0.5F, 0.5F, out, scratch);
    expect(state, std::array{2.0F, 4.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});
    expect(out, std::array{1.0F, 2.0F});

    // A normalized key spans the two active rows. The old state decays,
    // then the new delta contributes to both rows during the same step.
    const auto k = std::sqrt(0.5F);
    spark::delta_net_step(state, std::array{0.0F, 1.0F, 0.0F, 0.0F},
                         std::array{k, k, 0.0F, 0.0F}, std::array{3.0F * k, 6.0F * k},
                         0.5F, 0.5F, out, scratch);
    expect(state, std::array{1.5F, 3.0F, 0.5F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F});
    expect(out, std::array{0.25F, 0.5F});
    std::cout << "Gated DeltaNet recurrence tests passed\n";
}
