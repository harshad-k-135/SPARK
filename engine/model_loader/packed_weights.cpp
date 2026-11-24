#include "engine/model_loader/packed_weights.hpp"

#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace spark {
namespace {

using Bytes = std::span<const std::uint8_t>;

std::uint32_t word(Bytes bytes, std::size_t offset) {
    return std::uint32_t{bytes[offset]} | (std::uint32_t{bytes[offset + 1]} << 8);
}

float half(Bytes bytes, std::size_t offset) {
    static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4);
    const auto bits = word(bytes, offset);
    const auto exponent = (bits >> 10) & 31U;
    const auto fraction = bits & 1023U;
    const auto sign = (bits & 0x8000U) << 16;
    if (exponent == 0) {
        const auto magnitude = std::ldexp(static_cast<float>(fraction), -24);
        return sign == 0 ? magnitude : -magnitude;
    }
    const auto fp32_exponent = exponent == 31 ? 255U : exponent + 112U;
    return std::bit_cast<float>(sign | (fp32_exponent << 23) | (fraction << 13));
}

int signed_byte(std::uint8_t value) {
    return value < 128 ? value : static_cast<int>(value) - 256;
}

void unpack_k45(Bytes block, std::span<float> out, bool fifth_bit) {
    const auto scale = half(block, 0);
    const auto minimum = half(block, 2);
    const auto packed_scales = block.subspan(4, 12);
    const auto low_bits = block.subspan(fifth_bit ? 48 : 16);
    for (std::size_t group = 0; group < 8; ++group) {
        unsigned s;
        unsigned m;
        if (group < 4) {
            s = packed_scales[group] & 63U;
            m = packed_scales[group + 4] & 63U;
        } else {
            s = (packed_scales[group + 4] & 15U) | (packed_scales[group - 4] >> 6) << 4;
            m = (packed_scales[group + 4] >> 4) | (packed_scales[group] >> 6) << 4;
        }
        for (std::size_t i = 0; i < 32; ++i) {
            auto q = (low_bits[group / 2 * 32 + i] >> (4 * (group % 2))) & 15U;
            if (fifth_bit) {
                q |= ((block[16 + i] >> group) & 1U) << 4;
            }
            out[group * 32 + i] = (scale * static_cast<float>(s)) * static_cast<float>(q) -
                                 minimum * static_cast<float>(m);
        }
    }
}

void unpack_k6(Bytes block, std::span<float> out) {
    const auto scale = half(block, 208);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const auto section = i / 128;
        const auto quarter = i / 32 % 4;
        const auto lane = i % 32;
        const auto low = (block[section * 64 + quarter % 2 * 32 + lane] >>
                          (quarter / 2 * 4)) & 15;
        const auto high = (block[128 + section * 32 + lane] >> (quarter * 2)) & 3;
        const auto q = (low | (high << 4)) - 32;
        out[i] = (scale * static_cast<float>(signed_byte(block[192 + i / 16]))) *
                 static_cast<float>(q);
    }
}

}  // namespace

void unpack_gguf_row(std::uint32_t type, Bytes input, std::span<float> output) {
    const auto layout = gguf_encoding(type);
    if (output.empty() || output.size() % layout.elements != 0 ||
        input.size() % layout.bytes != 0 ||
        input.size() / layout.bytes != output.size() / layout.elements) {
        throw std::invalid_argument("packed GGUF row size does not match the output");
    }
    for (std::size_t start = 0; start < output.size(); start += layout.elements) {
        const auto block = input.subspan(start / layout.elements * layout.bytes, layout.bytes);
        const auto out = output.subspan(start, layout.elements);
        switch (type) {
            case 0:
                out[0] = std::bit_cast<float>(word(block, 0) | (word(block, 2) << 16));
                break;
            case 1: out[0] = half(block, 0); break;
            case 2: {
                const auto scale = half(block, 0);
                for (std::size_t i = 0; i < out.size(); ++i) {
                    const auto nibble = (block[2 + i % 16] >> (i / 16 * 4)) & 15;
                    out[i] = scale * static_cast<float>(nibble - 8);
                }
                break;
            }
            case 8: {
                const auto scale = half(block, 0);
                for (std::size_t i = 0; i < out.size(); ++i) {
                    out[i] = scale * static_cast<float>(signed_byte(block[2 + i]));
                }
                break;
            }
            case 12: unpack_k45(block, out, false); break;
            case 13: unpack_k45(block, out, true); break;
            case 14: unpack_k6(block, out); break;
            case 30: out[0] = std::bit_cast<float>(word(block, 0) << 16); break;
            default: throw std::logic_error("missing GGUF unpacking for a supported type");
        }
    }
}

void read_gguf_row(const std::filesystem::path& path, const GgufTensor& tensor,
                   std::uint64_t row, std::span<float> output) {
    if (output.size() != tensor.shape[0]) {
        throw std::invalid_argument("GGUF row width mismatch");
    }
    const auto layout = gguf_encoding(tensor.type);
    const auto row_bytes = (tensor.shape[0] / layout.elements) * layout.bytes;
    if (row_bytes == 0 || row >= tensor.bytes / row_bytes) {
        throw std::out_of_range("GGUF tensor row outside the payload");
    }
    std::ifstream input(path, std::ios::binary);
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(row_bytes));
    const auto offset = tensor.file_offset + row * row_bytes;
    if (!input.seekg(static_cast<std::streamoff>(offset)) ||
        !input.read(reinterpret_cast<char*>(packed.data()), static_cast<std::streamsize>(row_bytes))) {
        throw std::runtime_error("could not read GGUF tensor row");
    }
    unpack_gguf_row(tensor.type, packed, output);
}

}  // namespace spark
