#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <variant>

namespace spark {

struct GgufTensor {
    std::array<std::uint64_t, 4> shape{1, 1, 1, 1};
    std::uint32_t rank = 0;
    std::uint32_t type = 0;
    std::uint64_t file_offset = 0;
    std::uint64_t bytes = 0;
};

struct GgufArray {
    std::uint32_t element_type;
    std::uint64_t count;
    std::uint64_t file_offset;
};

using GgufValue = std::variant<std::uint64_t, std::int64_t, double, bool, std::string, GgufArray>;

class Gguf {
public:
    explicit Gguf(const std::filesystem::path& path);

    [[nodiscard]] const GgufValue& metadata(std::string_view key) const;
    [[nodiscard]] const auto& tensors() const { return tensors_; }
    [[nodiscard]] const auto& entries() const { return metadata_; }
    [[nodiscard]] std::uint64_t size() const { return size_; }
    [[nodiscard]] std::uint32_t version() const { return version_; }

private:
    std::map<std::string, GgufValue, std::less<>> metadata_;
    std::map<std::string, GgufTensor, std::less<>> tensors_;
    std::uint64_t size_ = 0;
    std::uint32_t version_ = 0;
};

[[nodiscard]] std::string_view gguf_type_name(std::uint32_t type);

}  // namespace spark
