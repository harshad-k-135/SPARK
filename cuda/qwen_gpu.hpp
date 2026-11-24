#pragma once

#include "engine/model_loader/gguf.hpp"
#include "engine/sampling/sampler.hpp"

#include <memory>

namespace spark {

void test_qwen_cuda_ops();

class GpuModel {
public:
    GpuModel(const Gguf& gguf, const std::filesystem::path& path, std::size_t context,
             bool profile = false);
    ~GpuModel();
    std::span<const float> step(TokenId token, bool logits);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace spark
