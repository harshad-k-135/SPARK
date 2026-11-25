#pragma once

#include "engine/model_loader/gguf.hpp"
#include "engine/sampling/sampler.hpp"

#include <memory>

namespace spark {

void test_qwen_cuda_ops();
enum class DecodeGraph { off, single, bucket };

class GpuModel {
public:
    GpuModel(const Gguf& gguf, const std::filesystem::path& path, std::size_t context,
             bool profile = false, DecodeGraph graph = DecodeGraph::off, bool packed = false,
             bool batch_prefill = false, bool q6_block_gemv = false,
             bool fp16_prefill = false, bool q4_block_gemv = false);
    ~GpuModel();
    void reset();
    std::span<const float> prefill(std::span<const TokenId> tokens);
    std::span<const float> step(TokenId token, bool logits);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace spark
