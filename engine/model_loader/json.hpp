#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace spark {

class Json {
public:
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json, std::less<>>;
    using Value = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;

    template <typename T>
    explicit Json(T value) : value_(std::in_place_type<T>, std::move(value)) {}
    [[nodiscard]] static Json read(const std::filesystem::path& path);
    [[nodiscard]] const Json& at(std::string_view key) const;
    [[nodiscard]] const Json* find(std::string_view key) const;
    [[nodiscard]] const Array& array() const;
    [[nodiscard]] const std::string& string() const;
    [[nodiscard]] double number() const;
    [[nodiscard]] std::size_t integer() const;
    [[nodiscard]] bool boolean() const;
    [[nodiscard]] bool is_array() const { return std::holds_alternative<Array>(value_); }

private:
    Value value_;
};

}  // namespace spark
