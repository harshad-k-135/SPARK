#include "engine/tokenizer/tokenizer.hpp"

#include <algorithm>
#include <utility>
#include <limits>
#include <stdexcept>

namespace spark {
namespace {

constexpr auto no_token = std::numeric_limits<TokenId>::max();

struct Range { std::uint32_t first, last, kind; };

constexpr Range ranges[] = {
#include "engine/tokenizer/unicode_ranges.inc"
};

std::uint32_t unicode_class(std::uint32_t codepoint) {
    const auto range = std::ranges::lower_bound(ranges, codepoint, {}, &Range::last);
    return range != std::end(ranges) && codepoint >= range->first ? range->kind : 0;
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
    bool word() const { return kind == 1 || kind == 4; }
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

std::unordered_map<std::uint32_t, unsigned char> byte_alphabet() {
    std::unordered_map<std::uint32_t, unsigned char> result;
    std::uint32_t extra = 256;
    for (std::uint32_t byte = 0; byte < 256; ++byte) {
        const bool direct = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                            (byte >= 174);
        result.emplace(direct ? byte : extra++, static_cast<unsigned char>(byte));
    }
    return result;
}

std::string decode_token_bytes(std::string_view token,
                               const std::unordered_map<std::uint32_t, unsigned char>& alphabet) {
    std::string result;
    for (std::size_t offset = 0; offset < token.size();) {
        const auto value = next_codepoint(token, offset);
        const auto entry = alphabet.find(value);
        if (entry == alphabet.end()) {
            throw std::runtime_error("Qwen3.5 vocabulary contains an invalid byte-alphabet symbol");
        }
        result.push_back(static_cast<char>(entry->second));
    }
    return result;
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
    if (first.kind != 1 && first.kind != 2 && !first.newline() &&
        begin + 1 < size && chars[begin + 1].word()) {
        ++end;
    }
    if (end < size && chars[end].word()) {
        do {
            ++end;
        } while (end < size && chars[end].word());
        return end;
    }
    if (first.kind == 2) {
        return begin + 1;
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
    throw std::logic_error("Qwen tokenizer split failed");
}

}  // namespace

Tokenizer::Tokenizer(const Gguf& model) {
    if (std::get<std::string>(model.metadata("tokenizer.ggml.model")) != "gpt2" ||
        std::get<std::string>(model.metadata("tokenizer.ggml.pre")) != "qwen35") {
        throw std::runtime_error("expected the Qwen3.5 byte BPE tokenizer");
    }
    const auto tokens = model.strings("tokenizer.ggml.tokens");
    const auto types = model.integers("tokenizer.ggml.token_type");
    if (tokens.size() != types.size() || tokens.size() >= no_token) {
        throw std::runtime_error("invalid Qwen3.5 vocabulary dimensions");
    }
    const auto alphabet = byte_alphabet();
    std::unordered_map<std::string_view, TokenId> lookup;
    vocabulary_.reserve(tokens.size());
    byte_tokens_.fill(no_token);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const auto id = static_cast<TokenId>(i);
        const bool special = types[i] != 1;
        if (special) {
            vocabulary_.push_back({tokens[i], true});
            if ((types[i] == 3 || types[i] == 4) && tokens[i].starts_with('<')) {
                special_tokens_.push_back(id);
            }
        } else {
            auto bytes = decode_token_bytes(tokens[i], alphabet);
            if (!lookup.emplace(tokens[i], id).second || bytes.empty()) {
                throw std::runtime_error("invalid Qwen3.5 vocabulary entry");
            }
            if (bytes.size() == 1) {
                byte_tokens_[static_cast<unsigned char>(bytes[0])] = id;
            }
            vocabulary_.push_back({std::move(bytes), false});
        }
    }
    if (std::ranges::find(byte_tokens_, no_token) != byte_tokens_.end()) {
        throw std::runtime_error("Qwen3.5 vocabulary lacks a byte token");
    }
    std::stable_sort(special_tokens_.begin(), special_tokens_.end(), [&](auto left, auto right) {
        return vocabulary_[left].bytes.size() > vocabulary_[right].bytes.size();
    });
    const auto merges = model.strings("tokenizer.ggml.merges");
    merges_.reserve(merges.size());
    for (std::size_t rank = 0; rank < merges.size(); ++rank) {
        const std::string_view merge = merges[rank];
        const auto separator = merge.find(' ');
        if (separator == std::string_view::npos) {
            throw std::runtime_error("invalid Qwen3.5 BPE merge");
        }
        const auto left = lookup.at(merge.substr(0, separator));
        const auto right = lookup.at(merge.substr(separator + 1));
        auto combined = std::string(merge.substr(0, separator));
        combined.append(merge.substr(separator + 1));
        const auto result = lookup.at(combined);
        if (!merges_.emplace(pair_key(left, right), Merge{result, static_cast<std::uint32_t>(rank)}).second) {
            throw std::runtime_error("repeated Qwen3.5 BPE merge");
        }
    }
}
void Tokenizer::encode_piece(std::string_view piece, std::vector<TokenId>& output) const {
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

std::vector<TokenId> Tokenizer::encode(std::string_view text, bool parse_special) const {
    std::vector<TokenId> output;
    if (!parse_special) {
        encode_ordinary(text, output);
        return output;
    }
    std::size_t begin = 0;
    for (auto marker = text.find('<'); marker != std::string_view::npos;
         marker = text.find('<', marker + 1)) {
        const auto match = std::ranges::find_if(special_tokens_, [&](auto id) {
            return text.substr(marker).starts_with(vocabulary_[id].bytes);
        });
        if (match != special_tokens_.end()) {
            encode_ordinary(text.substr(begin, marker - begin), output);
            output.push_back(*match);
            begin = marker + vocabulary_[*match].bytes.size();
            marker = begin - 1;
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
