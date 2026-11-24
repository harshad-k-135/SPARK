#include "engine/runtime/session.hpp"

#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <cwchar>
#endif

namespace {

constexpr auto usage =
    "Usage: spark generate --model DIR (--prompt TEXT | --prompt-file PATH) "
    "[--max-new-tokens N] [--temperature T] [--top-p P] [--seed N] [--context N]\n";

struct Options {
    std::filesystem::path model;
    std::string prompt;
    bool prompt_set = false;
    std::size_t context = 2048;
    std::size_t new_tokens = 64;
    spark::SamplingOptions sampling;
};

std::filesystem::path utf8_path(std::string_view value) {
    return std::filesystem::path(std::u8string(value.begin(), value.end()));
}

template <typename T>
T number(std::string_view value, std::string_view option) {
    T result{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("invalid value for " + std::string(option));
    }
    return result;
}

Options parse(std::span<const std::string> args) {
    if (args.empty() || args.front() != "generate") {
        throw std::invalid_argument(usage);
    }
    Options options;
    for (std::size_t i = 1; i < args.size(); ++i) {
        const auto& key = args[i];
        if (++i == args.size()) {
            throw std::invalid_argument("missing value for " + key);
        }
        const auto& value = args[i];
        if (key == "--model") {
            options.model = utf8_path(value);
        } else if (key == "--prompt" || key == "--prompt-file") {
            if (options.prompt_set) {
                throw std::invalid_argument("provide exactly one prompt");
            }
            options.prompt_set = true;
            if (key == "--prompt") {
                options.prompt = value;
            } else {
                std::ifstream input(utf8_path(value), std::ios::binary);
                if (!input) {
                    throw std::runtime_error("could not open prompt file");
                }
                options.prompt.assign(std::istreambuf_iterator<char>(input), {});
            }
        } else if (key == "--max-new-tokens") {
            options.new_tokens = number<std::size_t>(value, key);
        } else if (key == "--context") {
            options.context = number<std::size_t>(value, key);
        } else if (key == "--temperature") {
            options.sampling.temperature = number<double>(value, key);
        } else if (key == "--top-p") {
            options.sampling.top_p = number<double>(value, key);
        } else if (key == "--seed") {
            options.sampling.seed = number<std::uint64_t>(value, key);
        } else {
            throw std::invalid_argument("unknown option " + key);
        }
    }
    if (options.model.empty() || !options.prompt_set) {
        throw std::invalid_argument(usage);
    }
    return options;
}

int run(std::span<const std::string> args) {
    if (args.size() == 1 && args.front() == "--help") {
        std::cout << usage;
        return 0;
    }
    const auto options = parse(args);
    const spark::Tokenizer tokenizer(options.model / "tokenizer.bin");
    const spark::Model model(options.model);
    if (tokenizer.vocab_size() != model.config().vocab) {
        throw std::runtime_error("model and tokenizer vocabulary sizes differ");
    }
    auto prompt = tokenizer.encode(options.prompt);
    prompt.insert(prompt.begin(), model.config().bos);
    spark::Session session(model, options.context);
    const auto tokens = session.generate(prompt, options.new_tokens, options.sampling);
    std::cout << tokenizer.decode(tokens, true) << '\n';
    return 0;
}

#ifdef _WIN32
std::string argument(const wchar_t* text) {
    const auto length = static_cast<int>(std::wcslen(text));
    if (length == 0) {
        return {};
    }
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
#ifdef _WIN32
        if (_setmode(_fileno(stdout), _O_BINARY) == -1) {
            throw std::runtime_error("could not configure UTF-8 byte output");
        }
#endif
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
