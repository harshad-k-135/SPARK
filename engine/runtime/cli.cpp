#include "engine/model_loader/gguf.hpp"
#include "engine/model_loader/packed_weights.hpp"
#include "engine/tokenizer/tokenizer.hpp"
#include "engine/transformer/cpu_model.hpp"
#ifdef SPARK_WITH_CUDA
#include "cuda/qwen_gpu.hpp"
#endif

#include <iostream>
#include <cmath>
#include <algorithm>
#include <charconv>
#include <chrono>
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
#include <fcntl.h>
#include <io.h>
#endif

namespace {

constexpr auto usage =
    "Usage: spark inspect --model FILE.gguf [--tensors]\n"
    "       spark tokenize --model FILE.gguf --prompt TEXT [--special]\n"
    "       spark serve --model FILE.gguf\n"
    "       spark generate --model FILE.gguf (--prompt TEXT [--chat] | --prompt-ids IDS)\n"
    "              [--device cpu|cuda] [--max-new-tokens N] [--context N] [--temperature T]\n"
    "              [--top-p P] [--seed N]\n"
    "              [--ignore-eos] [--ids] [--first-logits PATH] [--profile] [--timings]\n"
    "              [--warmups N] [--trials N] [--cuda-graph off|single|bucket]\n"
    "              [--cuda-weights fp32|packed]\n";

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

std::vector<spark::TokenId> encode_prompt(const spark::Tokenizer& tokenizer,
                                          std::string_view text, bool chat) {
    auto input = tokenizer.encode(text);
    if (!chat) return input;
    auto prefix = tokenizer.encode("<|im_start|>user\n", true);
    const auto suffix = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", true);
    prefix.insert(prefix.end(), input.begin(), input.end());
    prefix.insert(prefix.end(), suffix.begin(), suffix.end());
    return prefix;
}

void serve(const std::filesystem::path& path) {
#ifdef SPARK_WITH_CUDA
    constexpr unsigned capacity = 2048;
    const spark::Gguf model(path);
    const spark::Tokenizer tokenizer(model);
    const auto eos = std::get<std::uint64_t>(model.metadata("tokenizer.ggml.eos_token_id"));
    const auto end = tokenizer.encode("<|endoftext|>", true);
    spark::GpuModel session(model, path, capacity, false, spark::DecodeGraph::off, true);
    std::cout << "{\"event\":\"ready\",\"device\":\"cuda\"}\n" << std::flush;
    unsigned length = 0, maximum = 0;
    while (std::cin >> length >> maximum) {
        if (std::cin.get() != '\n' || length == 0 || length > 32768 || maximum == 0 || maximum > 512)
            throw std::runtime_error("invalid worker request");
        std::string prompt(length, '\0');
        if (!std::cin.read(prompt.data(), length)) throw std::runtime_error("truncated worker request");
        std::vector<spark::TokenId> input;
        try {
            input = encode_prompt(tokenizer, prompt, true);
        } catch (const std::exception&) {
            std::cout << "{\"event\":\"error\",\"error\":\"Invalid UTF-8 prompt.\"}\n" << std::flush;
            continue;
        }
        if (input.size() > capacity - maximum) {
            std::cout << "{\"event\":\"error\",\"error\":\"Prompt and output exceed the 2,048-token context.\"}\n"
                      << std::flush;
            continue;
        }
        session.reset();
        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        auto first = start;
        auto last = start;
        std::span<const float> logits;
        for (std::size_t i = 0; i < input.size(); ++i)
            logits = session.step(input[i], i + 1 == input.size());
        spark::Sampler sampler(logits.size(), {});
        unsigned count = 0;
        for (; count < maximum; ++count) {
            const auto token = sampler.sample(logits);
            if (token == eos || (end.size() == 1 && token == end.front())) break;
            last = Clock::now();
            if (count == 0) first = last;
            const auto bytes = tokenizer.decode(std::span(&token, 1), true);
            constexpr auto hex = "0123456789abcdef";
            std::cout << "{\"event\":\"token\",\"bytes\":\"";
            for (const unsigned char byte : bytes) std::cout << hex[byte >> 4] << hex[byte & 15];
            std::cout << "\"}\n" << std::flush;
            if (count + 1 < maximum) logits = session.step(token, true);
        }
        const auto ttft = std::chrono::duration<double, std::milli>(first - start).count();
        const auto decode = std::chrono::duration<double>(last - first).count();
        std::cout << "{\"event\":\"done\",\"prompt_tokens\":" << input.size()
                  << ",\"generated_tokens\":" << count << ",\"first_token_ms\":" << ttft
                  << ",\"decode_tokens_per_sec\":" << (count > 1 && decode > 0 ? (count - 1) / decode : 0)
                  << "}\n" << std::flush;
    }
#else
    (void)path;
    throw std::runtime_error("this SPARK build has no CUDA worker");
#endif
}

void generate(const std::vector<std::string>& args) {
    if (args.size() < 5 || args[1] != "--model" ||
        (args[3] != "--prompt" && args[3] != "--prompt-ids"))
        throw std::invalid_argument(usage);
    bool chat = false, ids_only = false, ignore_eos = false, profile = false, timings = false;
    std::string_view device = "cpu";
    std::string_view graph = "off";
    std::string_view weights = "fp32";
    std::filesystem::path logits_path;
    std::size_t count = 64;
    std::size_t context = 2048;
    std::size_t warmups = 0, trials = 1;
    spark::SamplingOptions sampling;
    for (std::size_t i = 5; i < args.size(); ++i) {
        if (args[i] == "--chat") chat = true;
        else if (args[i] == "--ids") ids_only = true;
        else if (args[i] == "--ignore-eos") ignore_eos = true;
        else if (args[i] == "--profile") profile = true;
        else if (args[i] == "--timings") timings = true;
        else if (args[i] == "--device" && i + 1 < args.size()) device = args[++i];
        else if (args[i] == "--cuda-graph" && i + 1 < args.size()) graph = args[++i];
        else if (args[i] == "--cuda-weights" && i + 1 < args.size()) weights = args[++i];
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
        } else if (args[i] == "--warmups" && i + 1 < args.size()) {
            parse_number(args[i + 1], warmups, args[i]);
            ++i;
        } else if (args[i] == "--trials" && i + 1 < args.size()) {
            parse_number(args[i + 1], trials, args[i]);
            ++i;
        } else throw std::invalid_argument(usage);
    }
    if (context == 0 || context > 4096 || count == 0 || count > context)
        throw std::invalid_argument("requested token count or context is outside the supported range");
    if (ignore_eos && sampling.temperature != 0)
        throw std::invalid_argument("--ignore-eos requires greedy sampling");
    if (device != "cpu" && device != "cuda")
        throw std::invalid_argument("--device must be cpu or cuda");
    if (profile && device != "cuda")
        throw std::invalid_argument("--profile requires --device cuda");
    if (graph != "off" && ((graph != "single" && graph != "bucket") || device != "cuda" || profile))
        throw std::invalid_argument("CUDA graph requires a CUDA request without --profile");
    if (weights != "fp32" && (weights != "packed" || device != "cuda"))
        throw std::invalid_argument("--cuda-weights must be fp32 or packed on CUDA");
    if (trials == 0 || trials > 100 || warmups > 100)
        throw std::invalid_argument("invalid warmup or trial count");
    if ((warmups || trials != 1) && (!timings || !logits_path.empty()))
        throw std::invalid_argument("repeated runs require --timings and omit --first-logits");
    using Clock = std::chrono::steady_clock;
    const auto milliseconds = [](auto start, auto end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    };
    const auto load_start = Clock::now();
    const auto path = std::filesystem::path(std::u8string(args[2].begin(), args[2].end()));
    const spark::Gguf model(path);
    const spark::Tokenizer tokenizer(model);
    const bool use_ids = args[3] == "--prompt-ids";
    if (use_ids && chat) throw std::invalid_argument("--chat requires a text prompt");
    auto input = use_ids ? parse_ids(args[4]) : encode_prompt(tokenizer, args[4], chat);
    if (input.empty() || input.size() > context - count)
        throw std::invalid_argument("prompt and output exceed the context or prompt is empty");
    const auto eos = std::get<std::uint64_t>(model.metadata("tokenizer.ggml.eos_token_id"));
    const auto end_of_text = tokenizer.encode("<|endoftext|>", true);
    std::unique_ptr<spark::CpuModel> cpu;
#ifdef SPARK_WITH_CUDA
    std::unique_ptr<spark::GpuModel> gpu;
    if (device == "cuda") {
        const auto mode = graph == "single" ? spark::DecodeGraph::single
                        : graph == "bucket" ? spark::DecodeGraph::bucket : spark::DecodeGraph::off;
        gpu = std::make_unique<spark::GpuModel>(model, path, context, profile, mode,
                                               weights == "packed");
    }
#else
    if (device == "cuda") throw std::runtime_error("this SPARK build has no CUDA backend");
#endif
    if (device == "cpu") cpu = std::make_unique<spark::CpuModel>(model, path, context);
    const auto loaded = Clock::now();
    const auto step = [&](spark::TokenId token, bool emit) {
#ifdef SPARK_WITH_CUDA
        if (gpu) return gpu->step(token, emit);
#endif
        return cpu->step(token, emit);
    };
    const auto is_eos = [&](spark::TokenId token) {
        return token == eos || (end_of_text.size() == 1 && token == end_of_text.front());
    };
    const auto trial = [&](std::size_t run, bool warmup, bool display) {
        const auto prefill_start = Clock::now();
        std::span<const float> logits;
        for (std::size_t i = 0; i < input.size(); ++i)
            logits = step(input[i], i + 1 == input.size());
        const auto prefill_end = Clock::now();
        if (!logits_path.empty()) {
            std::ofstream file(logits_path, std::ios::binary);
            if (!file.write(reinterpret_cast<const char*>(logits.data()),
                            static_cast<std::streamsize>(logits.size_bytes())))
                throw std::runtime_error("could not write first-step logits");
        }
        spark::Sampler sampler(logits.size(), sampling);
        std::vector<spark::TokenId> output;
        output.reserve(count);
        auto first_token = prefill_end;
        auto last_token = prefill_end;
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
            if (timings) {
                last_token = Clock::now();
                if (output.empty()) first_token = last_token;
            }
            output.push_back(token);
            if (display && !ids_only)
                std::cout << tokenizer.decode(std::span(&token, 1), true) << std::flush;
            if (i + 1 < count) logits = step(token, true);
        }
        if (display) {
            if (ids_only) print_ids(output); else std::cout << '\n';
        }
        if (timings) {
            const auto prefill = milliseconds(prefill_start, prefill_end);
            const auto decode = milliseconds(first_token, last_token);
            std::cerr << "{\"event\":\"generation_timings\",\"device\":\"" << device
                      << "\",\"trial\":" << run << ",\"warmup\":" << (warmup ? "true" : "false")
                      << ",\"load_ms\":" << milliseconds(load_start, loaded)
                      << ",\"prefill_ms\":" << prefill
                      << ",\"first_token_ms\":" << milliseconds(prefill_start, first_token)
                      << ",\"decode_ms\":" << decode
                      << ",\"prompt_tokens\":" << input.size()
                      << ",\"generated_tokens\":" << output.size()
                      << ",\"prefill_tokens_per_sec\":" << (prefill > 0 ? input.size() * 1000.0 / prefill : 0)
                      << ",\"decode_tokens_per_sec\":"
                      << (output.size() > 1 && decode > 0 ? (output.size() - 1) * 1000.0 / decode : 0)
                      << "}\n";
        }
    };
    for (std::size_t i = 0; i < warmups + trials; ++i) {
        if (i > 0) {
#ifdef SPARK_WITH_CUDA
            if (gpu) gpu->reset(); else
#endif
            cpu->reset();
        }
        trial(i + 1, i < warmups, warmups + trials == 1);
    }
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
    if (args[0] == "serve" && args.size() == 3 && args[1] == "--model") {
#ifdef _WIN32
        if (_setmode(_fileno(stdin), _O_BINARY) < 0 || _setmode(_fileno(stdout), _O_BINARY) < 0)
            throw std::runtime_error("cannot configure worker pipe mode");
#endif
        serve(std::filesystem::path(std::u8string(args[2].begin(), args[2].end())));
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
