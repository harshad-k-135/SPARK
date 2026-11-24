#include "engine/sampling/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace spark {

Sampler::Sampler(std::size_t vocabulary, SamplingOptions options)
    : vocabulary_(vocabulary), options_(options), random_(options.seed) {
    if (vocabulary == 0 || vocabulary > std::numeric_limits<TokenId>::max() ||
        !std::isfinite(options.temperature) || options.temperature < 0 ||
        !std::isfinite(options.top_p) || options.top_p <= 0 || options.top_p > 1) {
        throw std::invalid_argument("invalid vocabulary, temperature, or top-p");
    }
    if (options.temperature > 0) {
        candidates_.resize(vocabulary);
    }
}

TokenId Sampler::sample(std::span<const float> logits) {
    if (logits.size() != vocabulary_ || !std::ranges::all_of(logits, [](float value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument("expected one finite logit per token");
    }
    const auto highest = std::ranges::max_element(logits);
    if (options_.temperature == 0) {
        return static_cast<TokenId>(highest - logits.begin());
    }
    double total = 0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const auto weight = std::exp((static_cast<double>(logits[i]) - *highest) /
                                     options_.temperature);
        candidates_[i] = {static_cast<TokenId>(i), weight};
        total += weight;
    }
    std::ranges::sort(candidates_, [](const auto& left, const auto& right) {
        return left.weight == right.weight ? left.id < right.id : left.weight > right.weight;
    });

    const auto cutoff = options_.top_p * total;
    double mass = 0;
    auto end = candidates_.begin();
    do {
        mass += end->weight;
        ++end;
    } while (mass < cutoff && end != candidates_.end() && end->weight > 0);

    const auto fraction = static_cast<double>(random_() >> 11) * 0x1.0p-53;
    auto draw = fraction * mass;
    auto candidate = candidates_.begin();
    while (candidate + 1 != end && draw >= candidate->weight) {
        draw -= candidate->weight;
        ++candidate;
    }
    return candidate->id;
}

}  // namespace spark
