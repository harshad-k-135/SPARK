#include "engine/model_loader/gguf.hpp"
#include "engine/model_loader/packed_weights.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cwchar>
#endif

namespace {

constexpr auto usage = "Usage: spark inspect --model FILE.gguf [--tensors]\n";

void inspect(const std::filesystem::path& path, bool list_tensors) {
    const spark::Gguf model(path);
    const auto& architecture = std::get<std::string>(model.metadata("general.architecture"));
    if (architecture != "qwen35") {
        throw std::runtime_error("expected a Qwen3.5 text GGUF (qwen35), found " + architecture);
    }
    const auto layers = std::get<std::uint64_t>(model.metadata("qwen35.block_count"));
    const auto hidden = std::get<std::uint64_t>(model.metadata("qwen35.embedding_length"));
    const auto& tokens = std::get<spark::GgufArray>(model.metadata("tokenizer.ggml.tokens"));
    if (layers == 0 || hidden == 0 || tokens.element_type != 8 || tokens.count == 0 ||
        model.tensors().empty()) {
        throw std::runtime_error("incomplete Qwen3.5 GGUF model");
    }

    std::uint64_t recurrent = 0;
    std::uint64_t attention = 0;
    std::uint64_t auxiliary = 0;
    const auto& tensors = model.tensors();
    for (std::uint64_t layer = 0; layer < layers; ++layer) {
        const auto prefix = "blk." + std::to_string(layer) + ".";
        const bool has_recurrent = tensors.contains(prefix + "ssm_a");
        const bool has_attention = tensors.contains(prefix + "attn_q.weight");
        if (has_recurrent == has_attention) {
            throw std::runtime_error("invalid Qwen3.5 layer type in " + prefix);
        }
        if (tensors.contains(prefix + "nextn.eh_proj.weight")) {
            ++auxiliary;
        } else if (has_recurrent) {
            ++recurrent;
        } else {
            ++attention;
        }
    }

    std::map<std::string_view, std::size_t> counts;
    for (const auto& [name, tensor] : model.tensors()) {
        ++counts[spark::gguf_encoding(tensor.type).name];
        if (list_tensors) {
            std::cout << name << '\t' << spark::gguf_encoding(tensor.type).name << '\t';
            for (std::uint32_t i = 0; i < tensor.rank; ++i) {
                if (i > 0) std::cout << 'x';
                std::cout << tensor.shape[i];
            }
            std::cout << "\toffset=" << tensor.file_offset << "\tbytes=" << tensor.bytes << '\n';
        }
    }
    std::cout << "architecture=" << architecture << "\ngguf_version=" << model.version()
              << "\nfile_blocks=" << layers << "\nrecurrent_layers=" << recurrent
              << "\nattention_layers=" << attention << "\nauxiliary_blocks=" << auxiliary
              << "\nhidden=" << hidden << "\nvocabulary=" << tokens.count
              << "\ntensors=" << model.tensors().size() << "\nfile_bytes=" << model.size() << '\n';
    for (const auto& [type, count] : counts) {
        std::cout << "tensors." << type << '=' << count << '\n';
    }
    const auto& embedding = tensors.at("token_embd.weight");
    if (embedding.rank != 2 || embedding.shape[0] != hidden || embedding.shape[1] != tokens.count) {
        throw std::runtime_error("Qwen3.5 embedding dimensions disagree with metadata");
    }
    std::vector<float> sample(hidden);
    spark::read_gguf_row(path, embedding, 0, sample);
    if (!std::ranges::all_of(sample, [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("non-finite values in the first embedding row");
    }
    const auto [lowest, highest] = std::ranges::minmax(sample);
    std::cout << "embedding_row_0_range=" << lowest << ',' << highest << '\n';
}

int run(const std::vector<std::string>& args) {
    if (args.empty() || (args.size() == 1 && args[0] == "--help")) {
        std::cout << usage;
        return 0;
    }
    if (args[0] == "generate") {
        throw std::runtime_error("native Qwen3.5 generation is pending CPU forward-pass parity");
    }
    if ((args.size() != 3 && args.size() != 4) || args[0] != "inspect" || args[1] != "--model" ||
        (args.size() == 4 && args[3] != "--tensors")) {
        throw std::invalid_argument(usage);
    }
    const auto path = std::filesystem::path(std::u8string(args[2].begin(), args[2].end()));
    inspect(path, args.size() == 4);
    return 0;
}

#ifdef _WIN32
std::string argument(const wchar_t* text) {
    const auto length = static_cast<int>(std::wcslen(text));
    if (length == 0) return {};
    const auto bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text, length,
                                           nullptr, 0, nullptr, nullptr);
    if (bytes == 0) {
        throw std::invalid_argument("argument could not be converted to UTF-8");
    }
    std::string result(static_cast<std::size_t>(bytes), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text, length,
                            result.data(), bytes, nullptr, nullptr) != bytes) {
        throw std::invalid_argument("argument could not be converted to UTF-8");
    }
    return result;
}
#else
std::string argument(const char* text) { return text; }
#endif

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        std::vector<std::string> args;
        for (int i = 1; i < argc; ++i) {
            args.push_back(argument(argv[i]));
        }
        return run(args);
    } catch (const std::exception& error) {
        std::cerr << "spark: " << error.what() << '\n';
        return 1;
    }
}
