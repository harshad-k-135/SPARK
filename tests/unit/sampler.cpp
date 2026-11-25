#include "engine/sampling/sampler.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_sampling() {
    const std::array<float, 5> logits{-1.f, 2.f, 2.f, -3.f, 1.f};
    spark::Sampler greedy(logits.size(), {});
    require(greedy.sample(logits) == 1, "greedy ties must select the lowest ID");
    try {
        auto invalid = logits;
        invalid.back() = std::numeric_limits<float>::quiet_NaN();
        (void)greedy.sample(invalid);
        throw std::runtime_error("greedy accepted a NaN");
    } catch (const std::invalid_argument&) {}

    const spark::SamplingOptions options{.temperature = 0.8f, .top_p = 0.7f, .seed = 772};
    spark::Sampler first(logits.size(), options), second(logits.size(), options);
    for (unsigned i = 0; i < 64; ++i) {
        const auto token = first.sample(logits);
        require(token == second.sample(logits), "seeded sampling diverged");
        require(token == 1 || token == 2, "top-p allowed a token outside the nucleus");
    }
}

}  // namespace

int main() {
    try {
        test_sampling();
        std::cout << "Sampler correctness tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
