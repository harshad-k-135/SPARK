#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace spark {

using TokenId = std::uint32_t;

class Tokenizer {
public:
    explicit Tokenizer(const std::filesystem::path& path);
    [[nodiscard]] std::vector<TokenId> encode(std::string_view text) const;
    [[nodiscard]] std::string decode(std::span<const TokenId> tokens,
                                     bool skip_special = false) const;
    [[nodiscard]] std::size_t vocab_size() const { return vocabulary_.size(); }

private:
    struct Entry {
        std::string bytes;
        bool special;
    };
    struct Merge {
        TokenId result;
        std::uint32_t rank;
    };
    struct Range {
        std::uint32_t first;
        std::uint32_t last;
        std::uint32_t kind;
    };

    [[nodiscard]] std::uint32_t unicode_class(std::uint32_t codepoint) const;
    void encode_ordinary(std::string_view text, std::vector<TokenId>& output) const;
    void encode_piece(std::string_view piece, std::vector<TokenId>& output) const;

    std::vector<Entry> vocabulary_;
    std::array<TokenId, 256> byte_tokens_{};
    std::vector<TokenId> special_tokens_;
    std::vector<Range> ranges_;
    std::unordered_map<std::string_view, TokenId> direct_tokens_;
    std::unordered_map<std::uint64_t, Merge> merges_;
    bool ignore_merges_ = false;

public:
    // The lookup table borrows strings from vocabulary_.
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
};

}  // namespace spark
