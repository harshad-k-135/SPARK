#pragma once

#include "engine/model_loader/model.hpp"

namespace spark {

class CpuForward {
public:
    CpuForward(const Model& model, std::size_t context);
    void push(TokenId token, std::size_t position);
    [[nodiscard]] std::span<const float> logits();

private:
    [[nodiscard]] std::span<float> cache_row(std::span<float> cache, std::size_t layer,
                                              std::size_t head, std::size_t position);
    void attention(std::size_t layer, std::size_t position);

    const Model& model_;
    std::size_t context_;
    std::vector<float> keys_;
    std::vector<float> values_;
    std::vector<float> cosine_;
    std::vector<float> sine_;
    std::vector<float> hidden_;
    std::vector<float> normalized_;
    std::vector<float> query_;
    std::vector<float> key_;
    std::vector<float> value_;
    std::vector<float> attended_;
    std::vector<float> projected_;
    std::vector<float> gate_;
    std::vector<float> up_;
    std::vector<float> scores_;
    std::vector<float> logits_;
};

}  // namespace spark
