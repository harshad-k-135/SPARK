#pragma once

#include "engine/model_loader/gguf.hpp"
#include "engine/sampling/sampler.hpp"

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace spark {

class Tokenizer {
public:
    explicit Tokenizer(const Gguf& model);
    [[nodiscard]] std::vector<TokenId> encode(std::string_view text,
                                             bool parse_special = false) const;
    [[nodiscard]] std::string decode(std::span<const TokenId> tokens,
                                     bool skip_special = false) const;
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

private:
    struct Entry {
        std::string bytes;
        bool special;
    };
    struct Merge {
        TokenId result;
        std::uint32_t rank;
    };

    void encode_ordinary(std::string_view text, std::vector<TokenId>& output) const;
    void encode_piece(std::string_view piece, std::vector<TokenId>& output) const;

    std::vector<Entry> vocabulary_;
    std::array<TokenId, 256> byte_tokens_{};
    std::vector<TokenId> special_tokens_;
    std::unordered_map<std::uint64_t, Merge> merges_;
};

}  // namespace spark
