#include "engine/model_loader/gguf.hpp"

#include <bit>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace spark {
namespace {

struct Encoding {
    std::string_view name;
    std::uint64_t elements;
    std::uint64_t bytes;
};

Encoding encoding(std::uint32_t type) {
    switch (type) {
        case 0: return {"F32", 1, 4};
        case 1: return {"F16", 1, 2};
        case 2: return {"Q4_0", 32, 18};
        case 8: return {"Q8_0", 32, 34};
        case 12: return {"Q4_K", 256, 144};
        case 13: return {"Q5_K", 256, 176};
        case 14: return {"Q6_K", 256, 210};
        case 30: return {"BF16", 1, 2};
        default: throw std::runtime_error("unsupported GGUF tensor type " + std::to_string(type));
    }
}

std::uint64_t multiply(std::uint64_t a, std::uint64_t b) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
        throw std::runtime_error("GGUF size overflow");
    }
    return a * b;
}

class Reader {
public:
    Reader(const std::filesystem::path& path, std::uint64_t size) : stream_(path, std::ios::binary),
                                                                   size_(size) {
        if (!stream_) {
            throw std::runtime_error("could not open GGUF file");
        }
    }

    template <typename T>
    T number() {
        static_assert(std::endian::native == std::endian::little);
        T value{};
        read(reinterpret_cast<char*>(&value), sizeof(value));
        return value;
    }

    std::string string() {
        const auto length = number<std::uint64_t>();
        if (length > remaining() || length > std::string{}.max_size()) {
            throw std::runtime_error("invalid GGUF string length");
        }
        std::string result(static_cast<std::size_t>(length), '\0');
        read(result.data(), length);
        return result;
    }

    void skip(std::uint64_t count) {
        if (count > remaining() ||
            !stream_.seekg(static_cast<std::streamoff>(count), std::ios::cur)) {
            throw std::runtime_error("invalid GGUF data range");
        }
        position_ += count;
    }

    std::uint64_t position() const { return position_; }
    std::uint64_t remaining() const { return size_ - position_; }

private:
    void read(char* out, std::uint64_t count) {
        if (count > remaining() ||
            !stream_.read(out, static_cast<std::streamsize>(count))) {
            throw std::runtime_error("truncated GGUF data");
        }
        position_ += count;
    }

    std::ifstream stream_;
    std::uint64_t size_;
    std::uint64_t position_ = 0;
};

std::uint64_t scalar_size(std::uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: throw std::runtime_error("unsupported GGUF metadata array element type");
    }
}

GgufValue read_value(Reader& reader, std::uint32_t type) {
    switch (type) {
        case 0: return std::uint64_t{reader.number<std::uint8_t>()};
        case 1: return std::int64_t{reader.number<std::int8_t>()};
        case 2: return std::uint64_t{reader.number<std::uint16_t>()};
        case 3: return std::int64_t{reader.number<std::int16_t>()};
        case 4: return std::uint64_t{reader.number<std::uint32_t>()};
        case 5: return std::int64_t{reader.number<std::int32_t>()};
        case 6: return double{reader.number<float>()};
        case 7: {
            const auto value = reader.number<std::uint8_t>();
            if (value > 1) {
                throw std::runtime_error("invalid GGUF boolean");
            }
            return value != 0;
        }
        case 8: return reader.string();
        case 9: {
            const auto element = reader.number<std::uint32_t>();
            const auto count = reader.number<std::uint64_t>();
            const GgufArray array{element, count, reader.position()};
            if (element == 8) {
                if (count > reader.remaining() / sizeof(std::uint64_t)) {
                    throw std::runtime_error("truncated GGUF string array");
                }
                for (std::uint64_t i = 0; i < count; ++i) {
                    reader.skip(reader.number<std::uint64_t>());
                }
            } else {
                reader.skip(multiply(count, scalar_size(element)));
            }
            return array;
        }
        case 10: return reader.number<std::uint64_t>();
        case 11: return reader.number<std::int64_t>();
        case 12: return reader.number<double>();
        default: throw std::runtime_error("invalid GGUF metadata value type");
    }
}

}  // namespace

Gguf::Gguf(const std::filesystem::path& path) : size_(std::filesystem::file_size(path)) {
    if (size_ > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("GGUF exceeds the supported file size");
    }
    Reader reader(path, size_);
    if (reader.number<std::uint32_t>() != 0x46554747U) {
        throw std::runtime_error("expected a little-endian GGUF file");
    }
    version_ = reader.number<std::uint32_t>();
    if (version_ != 3) {
        throw std::runtime_error("SPARK requires a GGUF v3 file");
    }
    const auto tensor_count = reader.number<std::uint64_t>();
    const auto metadata_count = reader.number<std::uint64_t>();
    if (tensor_count > reader.remaining() / 32 || metadata_count > reader.remaining() / 13) {
        throw std::runtime_error("invalid GGUF record counts");
    }
    for (std::uint64_t i = 0; i < metadata_count; ++i) {
        auto key = reader.string();
        const auto type = reader.number<std::uint32_t>();
        if (!metadata_.emplace(std::move(key), read_value(reader, type)).second) {
            throw std::runtime_error("duplicate GGUF metadata key");
        }
    }
    std::uint64_t alignment = 32;
    if (const auto item = metadata_.find("general.alignment"); item != metadata_.end()) {
        alignment = std::get<std::uint64_t>(item->second);
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::runtime_error("invalid GGUF data alignment");
    }

    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        auto name = reader.string();
        GgufTensor tensor;
        tensor.rank = reader.number<std::uint32_t>();
        if (tensor.rank == 0 || tensor.rank > tensor.shape.size()) {
            throw std::runtime_error("invalid GGUF rank in " + name);
        }
        for (std::uint32_t axis = 0; axis < tensor.rank; ++axis) {
            tensor.shape[axis] = reader.number<std::uint64_t>();
            if (tensor.shape[axis] == 0) {
                throw std::runtime_error("empty GGUF tensor axis in " + name);
            }
        }
        tensor.type = reader.number<std::uint32_t>();
        tensor.file_offset = reader.number<std::uint64_t>();
        const auto storage = encoding(tensor.type);
        if (tensor.file_offset % alignment != 0 || tensor.shape[0] % storage.elements != 0) {
            throw std::runtime_error("unaligned GGUF tensor in " + name);
        }
        tensor.bytes = multiply(tensor.shape[0] / storage.elements, storage.bytes);
        for (std::uint32_t axis = 1; axis < tensor.rank; ++axis) {
            tensor.bytes = multiply(tensor.bytes, tensor.shape[axis]);
        }
        if (!tensors_.emplace(std::move(name), tensor).second) {
            throw std::runtime_error("duplicate GGUF tensor name");
        }
    }
    const auto padding = (alignment - reader.position() % alignment) % alignment;
    reader.skip(padding);
    const auto data = reader.position();
    std::map<std::uint64_t, std::uint64_t> ranges;
    for (auto& [name, tensor] : tensors_) {
        if (tensor.file_offset > size_ - data || tensor.bytes > size_ - data - tensor.file_offset) {
            throw std::runtime_error("truncated GGUF tensor " + name);
        }
        tensor.file_offset += data;
        if (!ranges.emplace(tensor.file_offset, tensor.bytes).second) {
            throw std::runtime_error("overlapping GGUF tensor " + name);
        }
    }
    std::uint64_t end = data;
    for (const auto& [offset, bytes] : ranges) {
        if (offset < end) {
            throw std::runtime_error("overlapping GGUF tensors");
        }
        end = offset + bytes;
    }
}

const GgufValue& Gguf::metadata(std::string_view key) const {
    const auto entry = metadata_.find(key);
    if (entry == metadata_.end()) {
        throw std::runtime_error("missing GGUF metadata " + std::string(key));
    }
    return entry->second;
}

std::string_view gguf_type_name(std::uint32_t type) { return encoding(type).name; }

}  // namespace spark
