#include "engine/model_loader/json.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace spark {
namespace {

void append_utf8(std::string& text, unsigned cp) {
    if (cp < 0x80) {
        text.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        text.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        text.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        if (cp >= 0x10000) {
            text.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            text.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        } else {
            text.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        }
        text.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        text.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    Json parse() {
        auto result = value(0);
        whitespace();
        if (position_ != text_.size()) {
            fail("trailing data");
        }
        return result;
    }

private:
    [[noreturn]] void fail(std::string_view cause) const {
        throw std::runtime_error("JSON " + std::string(cause) + " at byte " +
                                 std::to_string(position_));
    }

    char peek() const { return position_ < text_.size() ? text_[position_] : '\0'; }

    void whitespace() {
        while (peek() == ' ' || peek() == '\n' || peek() == '\r' || peek() == '\t') {
            ++position_;
        }
    }

    bool match(char ch) {
        whitespace();
        if (peek() != ch) {
            return false;
        }
        ++position_;
        return true;
    }

    void require(char ch) {
        if (!match(ch)) {
            fail(std::string("expected ") + ch);
        }
    }

    void literal(std::string_view word) {
        if (!text_.substr(position_).starts_with(word)) {
            fail("invalid literal");
        }
        position_ += word.size();
    }

    unsigned hex_quad() {
        if (text_.size() - position_ < 4) {
            fail("truncated Unicode escape");
        }
        unsigned cp = 0;
        const auto* start = text_.data() + position_;
        const auto [end, error] = std::from_chars(start, start + 4, cp, 16);
        if (error != std::errc{} || end != start + 4) {
            fail("invalid Unicode escape");
        }
        position_ += 4;
        return cp;
    }

    std::string quoted() {
        require('"');
        std::string result;
        while (position_ < text_.size()) {
            const auto ch = text_[position_++];
            if (ch == '"') {
                return result;
            }
            if (static_cast<unsigned char>(ch) < 0x20) {
                fail("invalid control character");
            }
            if (ch != '\\') {
                result.push_back(ch);
                continue;
            }
            if (position_ == text_.size()) {
                fail("truncated escape");
            }
            switch (text_[position_++]) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': {
                    auto cp = hex_quad();
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        literal("\\u");
                        const auto low = hex_quad();
                        if (low < 0xDC00 || low > 0xDFFF) {
                            fail("invalid surrogate pair");
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + low - 0xDC00;
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        fail("unpaired surrogate");
                    }
                    append_utf8(result, cp);
                    break;
                }
                default: fail("unsupported escape");
            }
        }
        fail("unterminated string");
    }

    double number() {
        const auto* start = text_.data() + position_;
        double result = 0;
        const auto [end, error] = std::from_chars(start, text_.data() + text_.size(), result);
        if (error != std::errc{} || !std::isfinite(result)) {
            fail("invalid number");
        }
        position_ += static_cast<std::size_t>(end - start);
        return result;
    }

    Json value(unsigned depth) {
        if (depth > 64) {
            fail("nesting limit exceeded");
        }
        whitespace();
        switch (peek()) {
            case '{': {
                ++position_;
                Json::Object result;
                if (!match('}')) {
                    do {
                        auto key = quoted();
                        require(':');
                        if (!result.emplace(std::move(key), value(depth + 1)).second) {
                            fail("duplicate key");
                        }
                    } while (match(','));
                    require('}');
                }
                return Json(std::move(result));
            }
            case '[': {
                ++position_;
                Json::Array result;
                if (!match(']')) {
                    do {
                        result.push_back(value(depth + 1));
                    } while (match(','));
                    require(']');
                }
                return Json(std::move(result));
            }
            case '"': return Json(quoted());
            case 't': literal("true"); return Json(true);
            case 'f': literal("false"); return Json(false);
            case 'n': literal("null"); return Json(nullptr);
            default: return Json(number());
        }
    }

    std::string_view text_;
    std::size_t position_ = 0;
};

}  // namespace

Json Json::read(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open JSON file " + path.string());
    }
    const std::string text{std::istreambuf_iterator<char>(input), {}};
    return Parser(text).parse();
}

const Json* Json::find(std::string_view key) const {
    const auto& object = std::get<Object>(value_);
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
}

const Json& Json::at(std::string_view key) const {
    if (const auto* value = find(key)) {
        return *value;
    }
    throw std::runtime_error("missing JSON key " + std::string(key));
}

const Json::Array& Json::array() const { return std::get<Array>(value_); }
const std::string& Json::string() const { return std::get<std::string>(value_); }
double Json::number() const { return std::get<double>(value_); }
bool Json::boolean() const { return std::get<bool>(value_); }

std::size_t Json::integer() const {
    const auto value = number();
    constexpr auto max_integer = std::min(9007199254740991.0,
        static_cast<double>(std::numeric_limits<std::size_t>::max()));
    if (value < 0 || value > max_integer || std::trunc(value) != value) {
        throw std::runtime_error("invalid nonnegative JSON integer");
    }
    return static_cast<std::size_t>(value);
}

}  // namespace spark
