#include "engine/runtime/session.hpp"

#include <algorithm>
#include <stdexcept>

namespace spark {

Session::Session(const Model& model, std::size_t context)
    : model_(model), context_(context), forward_(model, context) {}

std::span<const float> Session::prefill(std::span<const TokenId> tokens) {
    if (tokens.empty() || tokens.size() > context_) {
        throw std::invalid_argument("prompt token count must be between one and the context limit");
    }
    if (std::ranges::any_of(tokens, [&](auto token) { return token >= model_.config().vocab; })) {
        throw std::invalid_argument("prompt token is outside the model vocabulary");
    }
    position_ = 0;
    for (const auto token : tokens) {
        forward_.push(token, position_++);
    }
    return forward_.logits();
}

std::span<const float> Session::decode_step(TokenId token) {
    if (position_ == 0 || position_ >= context_) {
        throw std::logic_error("decode requires a prefilled session with room in the context");
    }
    if (token >= model_.config().vocab) {
        throw std::invalid_argument("decode token is outside the model vocabulary");
    }
    forward_.push(token, position_++);
    return forward_.logits();
}

std::vector<TokenId> Session::generate(std::span<const TokenId> prompt, std::size_t new_tokens,
                                       SamplingOptions options) {
    if (prompt.size() > context_ || new_tokens > context_ - prompt.size()) {
        throw std::invalid_argument("prompt plus requested output exceeds the context limit");
    }
    Sampler sampler(model_.config().vocab, options);
    std::vector<TokenId> output;
    output.reserve(new_tokens);
    if (new_tokens == 0) {
        return output;
    }
    auto logits = prefill(prompt);
    const auto& eos = model_.config().eos;
    for (std::size_t step = 0; step < new_tokens; ++step) {
        const auto token = sampler.sample(logits);
        if (std::ranges::find(eos, token) != eos.end()) {
            break;
        }
        output.push_back(token);
        if (step + 1 < new_tokens) {
            logits = decode_step(token);
        }
    }
    return output;
}

}  // namespace spark
