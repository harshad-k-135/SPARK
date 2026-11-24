#pragma once

#include "engine/transformer/cpu_forward.hpp"

namespace spark {

class Session {
public:
    explicit Session(const Model& model, std::size_t context = 2048);
    [[nodiscard]] std::span<const float> prefill(std::span<const TokenId> tokens);
    [[nodiscard]] std::span<const float> decode_step(TokenId token);
    [[nodiscard]] std::size_t context() const { return context_; }

private:
    const Model& model_;
    std::size_t context_;
    std::size_t position_ = 0;
    CpuForward forward_;
};

}  // namespace spark
