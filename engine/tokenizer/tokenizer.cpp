#include "engine/tokenizer/tokenizer.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace spark {
namespace {

constexpr auto no_token = std::numeric_limits<TokenId>::max();

std::uint32_t read_u32(std::istream& input) {
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift != 32; shift += 8) {
        const auto byte = input.get();
        if (byte == std::char_traits<char>::eof()) {
            throw std::runtime_error("truncated SPARK tokenizer table");
        }
        value |= static_cast<std::uint32_t>(byte) << shift;
    }
    return value;
}

std::uint64_t pair_key(TokenId left, TokenId right) {
    return (static_cast<std::uint64_t>(left) << 32) | right;
}

struct Character {
    std::uint32_t codepoint;
    std::uint32_t kind;
    std::size_t offset;

    bool newline() const { return codepoint == '\r' || codepoint == '\n'; }
    bool space() const { return kind == 3; }
    bool symbol() const { return kind == 0; }
};

std::uint32_t next_codepoint(std::string_view text, std::size_t& offset) {
    const auto first = static_cast<unsigned char>(text[offset++]);
    if (first < 0x80) {
        return first;
    }
    if (first < 0xC2 || first > 0xF4) {
        throw std::runtime_error("prompt is not valid UTF-8");
    }
    const unsigned length = first < 0xE0 ? 2 : first < 0xF0 ? 3 : 4;
    std::uint32_t value = first & (0x7F >> length);
    for (unsigned i = 1; i < length; ++i) {
        if (offset == text.size()) {
            throw std::runtime_error("truncated UTF-8 prompt");
        }
        const auto byte = static_cast<unsigned char>(text[offset++]);
        if ((byte & 0xC0) != 0x80) {
            throw std::runtime_error("prompt is not valid UTF-8");
        }
        value = (value << 6) | (byte & 0x3F);
    }
    if ((length == 3 && value < 0x800) || (length == 4 && value < 0x10000) ||
        (value >= 0xD800 && value <= 0xDFFF) || value > 0x10FFFF) {
        throw std::runtime_error("prompt is not valid UTF-8");
    }
    return value;
}

std::uint32_t contraction_lower(std::uint32_t value) {
    if (value == 0x17F) {
        return 's';
    }
    return value >= 'A' && value <= 'Z' ? value + ('a' - 'A') : value;
}

std::size_t piece_end(std::span<const Character> chars, std::size_t begin) {
    const auto& first = chars[begin];
    const auto size = chars.size();
    if (first.codepoint == '\'') {
        for (const std::string_view ending : {"s", "t", "re", "ve", "m", "ll", "d"}) {
            if (begin + 1 + ending.size() > size) {
                continue;
            }
            const auto tail = chars.subspan(begin + 1, ending.size());
            if (std::equal(tail.begin(), tail.end(), ending.begin(), [](auto ch, char letter) {
                    return contraction_lower(ch.codepoint) == static_cast<unsigned char>(letter);
                })) {
                return begin + 1 + ending.size();
            }
        }
    }

    auto end = begin;
    if (first.kind != 1 && first.kind != 2 && !first.newline()) {
        ++end;
    }
    if (end < size && chars[end].kind == 1) {
        do {
            ++end;
        } while (end < size && chars[end].kind == 1);
        return end;
    }
    if (first.kind == 2) {
        end = begin + 1;
        while (end < size && end - begin < 3 && chars[end].kind == 2) {
            ++end;
        }
        return end;
    }

    end = begin + (first.codepoint == ' ' ? 1 : 0);
    if (end < size && chars[end].symbol()) {
        do {
            ++end;
        } while (end < size && chars[end].symbol());
        while (end < size && chars[end].newline()) {
            ++end;
        }
        return end;
    }

    end = begin;
    auto after_newline = begin;
    while (end < size && chars[end].space()) {
        ++end;
        if (chars[end - 1].newline()) {
            after_newline = end;
        }
    }
    if (after_newline != begin) {
        return after_newline;
    }
    if (end == size || end == begin + 1) {
        return end;
    }
    if (end > begin + 1) {
        return end - 1;
    }
    throw std::logic_error("Llama tokenizer split failed");
}

}  // namespace

Tokenizer::Tokenizer(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::array<char, 8> magic{};
    if (!input.read(magic.data(), magic.size()) ||
        std::string_view(magic.data(), magic.size()) != "SPKTT01\n") {
        throw std::runtime_error("invalid SPARK tokenizer table");
    }
    const auto count = read_u32(input);
    const auto merge_count = read_u32(input);
    const auto range_count = read_u32(input);
    const auto ignore_merges = read_u32(input);
    if (count < 256 || count > 1'000'000 || merge_count > 5'000'000 ||
        range_count > 10'000 || ignore_merges > 1) {
        throw std::runtime_error("invalid SPARK tokenizer table dimensions");
    }
    ignore_merges_ = ignore_merges != 0;
    vocabulary_.reserve(count);
    byte_tokens_.fill(no_token);
    for (TokenId id = 0; id < count; ++id) {
        const auto length = read_u32(input);
        const auto flags = read_u32(input);
        if (length == 0 || length > 4'096 || flags > 1) {
            throw std::runtime_error("invalid SPARK tokenizer vocabulary entry");
        }
        std::string bytes(length, '\0');
        if (!input.read(bytes.data(), static_cast<std::streamsize>(length))) {
            throw std::runtime_error("truncated SPARK tokenizer vocabulary");
        }
        vocabulary_.push_back({std::move(bytes), flags == 1});
    }
    for (TokenId id = 0; id < count; ++id) {
        const auto& entry = vocabulary_[id];
        if (entry.special) {
            special_tokens_.push_back(id);
        } else {
            if (!direct_tokens_.emplace(entry.bytes, id).second) {
                throw std::runtime_error("duplicate SPARK vocabulary bytes");
            }
            if (entry.bytes.size() == 1) {
                byte_tokens_[static_cast<unsigned char>(entry.bytes[0])] = id;
            }
        }
    }
    if (std::ranges::find(byte_tokens_, no_token) != byte_tokens_.end()) {
        throw std::runtime_error("SPARK vocabulary is missing a byte token");
    }
    std::stable_sort(special_tokens_.begin(), special_tokens_.end(), [&](auto left, auto right) {
        return vocabulary_[left].bytes.size() > vocabulary_[right].bytes.size();
    });
    merges_.reserve(merge_count);
    for (std::uint32_t rank = 0; rank < merge_count; ++rank) {
        const auto left = read_u32(input);
        const auto right = read_u32(input);
        const auto result = read_u32(input);
        if (std::max({left, right, result}) >= count ||
            !merges_.emplace(pair_key(left, right), Merge{result, rank}).second) {
            throw std::runtime_error("invalid BPE merge entry");
        }
    }
    for (std::uint32_t i = 0; i < range_count; ++i) {
        const Range range{read_u32(input), read_u32(input), read_u32(input)};
        if (range.first > range.last || range.last > 0x10FFFF || range.kind < 1 ||
            range.kind > 3 || (!ranges_.empty() && range.first <= ranges_.back().last)) {
            throw std::runtime_error("invalid tokenizer Unicode range");
        }
        ranges_.push_back(range);
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("trailing data in SPARK tokenizer table");
    }
}

std::uint32_t Tokenizer::unicode_class(std::uint32_t codepoint) const {
    const auto range = std::ranges::lower_bound(ranges_, codepoint, {}, &Range::last);
    return range != ranges_.end() && codepoint >= range->first ? range->kind : 0;
}

void Tokenizer::encode_piece(std::string_view piece, std::vector<TokenId>& output) const {
    if (ignore_merges_) {
        if (const auto token = direct_tokens_.find(piece); token != direct_tokens_.end()) {
            output.push_back(token->second);
            return;
        }
    }
    std::vector<TokenId> tokens;
    tokens.reserve(piece.size());
    for (unsigned char byte : piece) {
        tokens.push_back(byte_tokens_[byte]);
    }
    while (tokens.size() > 1) {
        auto best_rank = no_token;
        auto best_index = tokens.size();
        TokenId merged = 0;
        for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
            const auto match = merges_.find(pair_key(tokens[i], tokens[i + 1]));
            if (match != merges_.end() && match->second.rank < best_rank) {
                best_rank = match->second.rank;
                best_index = i;
                merged = match->second.result;
            }
        }
        if (best_index == tokens.size()) {
            break;
        }
        tokens[best_index] = merged;
        tokens.erase(tokens.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
    }
    output.insert(output.end(), tokens.begin(), tokens.end());
}

void Tokenizer::encode_ordinary(std::string_view text, std::vector<TokenId>& output) const {
    std::vector<Character> chars;
    for (std::size_t offset = 0; offset < text.size();) {
        const auto begin = offset;
        const auto cp = next_codepoint(text, offset);
        chars.push_back({cp, unicode_class(cp), begin});
    }
    for (std::size_t begin = 0; begin < chars.size();) {
        const auto end = piece_end(chars, begin);
        const auto byte_end = end == chars.size() ? text.size() : chars[end].offset;
        encode_piece(text.substr(chars[begin].offset, byte_end - chars[begin].offset), output);
        begin = end;
    }
}

std::vector<TokenId> Tokenizer::encode(std::string_view text) const {
    std::vector<TokenId> output;
    std::size_t begin = 0;
    for (auto marker = text.find("<|"); marker != std::string_view::npos;
         marker = text.find("<|", marker + 2)) {
        const auto match = std::ranges::find_if(special_tokens_, [&](auto id) {
            return text.substr(marker).starts_with(vocabulary_[id].bytes);
        });
        if (match != special_tokens_.end()) {
            encode_ordinary(text.substr(begin, marker - begin), output);
            output.push_back(*match);
            begin = marker + vocabulary_[*match].bytes.size();
            marker = begin - 2;
        }
    }
    encode_ordinary(text.substr(begin), output);
    return output;
}

std::string Tokenizer::decode(std::span<const TokenId> tokens, bool skip_special) const {
    std::string result;
    for (auto id : tokens) {
        const auto& entry = vocabulary_.at(id);
        if (!skip_special || !entry.special) {
            result.append(entry.bytes);
        }
    }
    return result;
}

}  // namespace spark
