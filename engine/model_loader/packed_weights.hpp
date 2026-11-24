#pragma once

#include "engine/model_loader/gguf.hpp"

#include <span>

namespace spark {

void unpack_gguf_row(std::uint32_t type, std::span<const std::uint8_t> input,
                     std::span<float> output);

void read_gguf_row(const std::filesystem::path& path, const GgufTensor& tensor,
                   std::uint64_t row, std::span<float> output);

}  // namespace spark
