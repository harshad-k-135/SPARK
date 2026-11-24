#include "engine/model_loader/gguf.hpp"
#include "engine/model_loader/packed_weights.hpp"
#include "engine/tokenizer/tokenizer.hpp"
#include "engine/transformer/cpu_model.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
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

constexpr auto usage =
    "Usage: spark inspect --model FILE.gguf [--tensors]\n"
    "       spark tokenize --model FILE.gguf --prompt TEXT [--special]\n"
    "       spark generate --model FILE.gguf (--prompt TEXT [--chat] | --prompt-ids IDS)\n"
    "              [--max-new-tokens N] [--context N] [--temperature T] [--top-p P] [--seed N]\n"
    "              [--ignore-eos] [--ids] [--first-logits PATH]\n";

template <typename T>
void parse_number(std::string_view value, T& result, std::string_view option) {
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size())
        throw std::invalid_argument("invalid value for " + std::string(option));
}

std::vector<spark::TokenId> parse_ids(std::string_view input) {
    std::vector<spark::TokenId> result;
    while (!input.empty()) {
        const auto delimiter = input.find(',');
        const auto item = input.substr(0, delimiter);
        spark::TokenId id{};
        const auto [end, error] = std::from_chars(item.data(), item.data() + item.size(), id);
        if (error != std::errc{} || end != item.data() + item.size() ||
            (delimiter != std::string_view::npos && delimiter + 1 == input.size()))
            throw std::invalid_argument("expected comma-separated prompt token IDs");
        result.push_back(id);
        if (delimiter == std::string_view::npos) break;
        input.remove_prefix(delimiter + 1);
    }
    return result;
}

void print_ids(std::span<const spark::TokenId> ids) {
    std::cout << '[';
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) std::cout << ',';
        std::cout << ids[i];
    }
    std::cout << "]\n";
}

void generate(const std::vector<std::string>& args) {
    if (args.size() < 5 || args[1] != "--model" ||
        (args[3] != "--prompt" && args[3] != "--prompt-ids"))
        throw std::invalid_argument(usage);
    bool chat = false, ids_only = false, ignore_eos = false;
    std::filesystem::path logits_path;
    std::size_t count = 64;
    std::size_t context = 2048;
    spark::SamplingOptions sampling;
    for (std::size_t i = 5; i < args.size(); ++i) {
        if (args[i] == "--chat") chat = true;
        else if (args[i] == "--ids") ids_only = true;
        else if (args[i] == "--ignore-eos") ignore_eos = true;
        else if (args[i] == "--first-logits" && i + 1 < args.size()) {
            const auto& value = args[++i];
            logits_path = std::filesystem::path(std::u8string(value.begin(), value.end()));
        } else if (args[i] == "--max-new-tokens" && i + 1 < args.size()) {
            parse_number(args[i + 1], count, args[i]);
            ++i;
        } else if (args[i] == "--context" && i + 1 < args.size()) {
            parse_number(args[i + 1], context, args[i]);
            ++i;
        } else if (args[i] == "--temperature" && i + 1 < args.size()) {
            parse_number(args[i + 1], sampling.temperature, args[i]);
            ++i;
        } else if (args[i] == "--top-p" && i + 1 < args.size()) {
            parse_number(args[i + 1], sampling.top_p, args[i]);
            ++i;
        } else if (args[i] == "--seed" && i + 1 < args.size()) {
            parse_number(args[i + 1], sampling.seed, args[i]);
            ++i;
        } else throw std::invalid_argument(usage);
    }
    if (context == 0 || context > 4096 || count == 0 || count > context)
        throw std::invalid_argument("requested token count or context is outside the supported range");
    if (ignore_eos && sampling.temperature != 0)
        throw std::invalid_argument("--ignore-eos requires greedy sampling");
    const auto path = std::filesystem::path(std::u8string(args[2].begin(), args[2].end()));
    const spark::Gguf model(path);
    const spark::Tokenizer tokenizer(model);
    const bool use_ids = args[3] == "--prompt-ids";
    if (use_ids && chat) throw std::invalid_argument("--chat requires a text prompt");
    auto input = use_ids ? parse_ids(args[4]) : tokenizer.encode(args[4]);
    if (chat) {
        auto prefix = tokenizer.encode("<|im_start|>user\n", true);
        const auto suffix = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", true);
        prefix.insert(prefix.end(), input.begin(), input.end());
        prefix.insert(prefix.end(), suffix.begin(), suffix.end());
        input = std::move(prefix);
    }
    if (input.empty() || input.size() > context - count)
        throw std::invalid_argument("prompt and output exceed the context or prompt is empty");
    const auto eos = std::get<std::uint64_t>(model.metadata("tokenizer.ggml.eos_token_id"));
    const auto end_of_text = tokenizer.encode("<|endoftext|>", true);
    spark::CpuModel session(model, path, context);
    std::span<const float> logits;
    for (std::size_t i = 0; i < input.size(); ++i)
        logits = session.step(input[i], i + 1 == input.size());
    if (!logits_path.empty()) {
        std::ofstream file(logits_path, std::ios::binary);
        if (!file.write(reinterpret_cast<const char*>(logits.data()),
                        static_cast<std::streamsize>(logits.size_bytes())))
            throw std::runtime_error("could not write first-step logits");
    }
    const auto is_eos = [&](spark::TokenId token) {
        return token == eos || (end_of_text.size() == 1 && token == end_of_text.front());
    };
    spark::Sampler sampler(logits.size(), sampling);
    std::vector<spark::TokenId> output;
    output.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        auto token = sampler.sample(logits);
        if (is_eos(token)) {
            if (!ignore_eos) break;
            auto best = -std::numeric_limits<float>::infinity();
            for (std::size_t j = 0; j < logits.size(); ++j) {
                const auto candidate = static_cast<spark::TokenId>(j);
                if (!is_eos(candidate) && logits[j] > best) {
                    best = logits[j];
                    token = candidate;
                }
            }
        }
        output.push_back(token);
        if (!ids_only) std::cout << tokenizer.decode(std::span(&token, 1), true) << std::flush;
        if (i + 1 < count) logits = session.step(token, true);
    }
    if (ids_only) print_ids(output); else std::cout << '\n';
}

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
        generate(args);
        return 0;
    }
    if ((args.size() == 5 || args.size() == 6) && args[0] == "tokenize" && args[1] == "--model" &&
        args[3] == "--prompt" && (args.size() == 5 || args[5] == "--special")) {
        const auto path = std::filesystem::path(std::u8string(args[2].begin(), args[2].end()));
        const spark::Gguf model(path);
        const spark::Tokenizer tokenizer(model);
        const auto ids = tokenizer.encode(args[4], args.size() == 6);
        if (tokenizer.decode(ids) != args[4]) {
            throw std::runtime_error("Qwen3.5 tokenizer round-trip changed the prompt bytes");
        }
        print_ids(ids);
        return 0;
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
