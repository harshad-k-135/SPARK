#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace spark {

using TokenId = std::uint32_t;

struct SamplingOptions {
    double temperature = 0.0;
    double top_p = 1.0;
    std::uint64_t seed = 0;
};

class Sampler {
public:
    Sampler(std::size_t vocabulary, SamplingOptions options);
    [[nodiscard]] TokenId sample(std::span<const float> logits);

private:
    struct Candidate {
        TokenId id;
        double weight;
    };

    std::size_t vocabulary_;
    SamplingOptions options_;
    std::mt19937_64 random_;
    std::vector<Candidate> candidates_;
};

}  // namespace spark
