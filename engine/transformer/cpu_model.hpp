#pragma once
#include "engine/model_loader/gguf.hpp"
#include "engine/sampling/sampler.hpp"
#include <memory>

namespace spark {

class CpuModel {
public:
    CpuModel(const Gguf& gguf, const std::filesystem::path& path, std::size_t context);
    ~CpuModel();
    // Consume one token; request logits only at generation boundaries.
    std::span<const float> step(TokenId token, bool logits);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace spark
