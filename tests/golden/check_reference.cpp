#include "engine/model_loader/json.hpp"
#include "engine/runtime/session.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>

namespace {

constexpr auto model_id = "meta-llama/Llama-3.2-1B-Instruct";
constexpr auto revision = "9213176726f574b556790deb65791e0c5aa438b6";

std::vector<spark::TokenId> ids(const spark::Json& data, std::size_t vocabulary) {
    std::vector<spark::TokenId> tokens;
    tokens.reserve(data.array().size());
    for (const auto& value : data.array()) {
        const auto id = value.integer();
        if (id >= vocabulary || id > std::numeric_limits<spark::TokenId>::max()) {
            throw std::runtime_error("reference contains a token outside the vocabulary");
        }
        tokens.push_back(static_cast<spark::TokenId>(id));
    }
    return tokens;
}

int check(const std::filesystem::path& model_dir, const std::filesystem::path& reference_path,
          bool official) {
    if (!std::filesystem::exists(reference_path)) {
        throw std::runtime_error("missing reference JSON; run tools/reference/generate_reference.py");
    }
    const auto reference = spark::Json::read(reference_path);
    if (reference.at("format").string() != "spark-golden-v1" ||
        reference.at("dtype").string() != "float32") {
        throw std::runtime_error("expected a spark-golden-v1 FP32 reference");
    }
    if (official && (reference.at("model_id").string() != model_id ||
                     reference.at("revision").string() != revision)) {
        throw std::runtime_error("Gate 1 requires the pinned Meta Llama 3.2 1B Instruct revision");
    }
    const spark::Model model(model_dir);
    const spark::Tokenizer tokenizer(model_dir / "tokenizer.bin");
    const auto& config = model.config();
    if (model.model_id() != reference.at("model_id").string() ||
        model.revision() != reference.at("revision").string() ||
        tokenizer.vocab_size() != config.vocab) {
        throw std::runtime_error("model, tokenizer, and reference metadata disagree");
    }
    spark::Session session(model, std::min<std::size_t>(2048, config.max_positions));
    spark::Sampler greedy(config.vocab, {});
    const auto& cases = reference.at("cases").array();
    if (cases.empty()) {
        throw std::runtime_error("reference has no cases");
    }

    std::size_t top_matches = 0;
    std::size_t matched_sequences = 0;
    std::size_t matched_tokens = 0;
    std::size_t total_tokens = 0;
    std::size_t shortest = std::numeric_limits<std::size_t>::max();
    std::size_t longest = 0;
    double worst_error = 0;
    std::set<std::string> names;
    for (const auto& entry : cases) {
        const auto& name = entry.at("id").string();
        if (name.empty() || name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-") !=
                            std::string::npos || !names.insert(name).second) {
            throw std::runtime_error("invalid or repeated reference case ID");
        }
        const auto& prompt = entry.at("prompt").string();
        const auto expected_prompt = ids(entry.at("prompt_ids"), config.vocab);
        const auto expected_output = ids(entry.at("generated_ids"), config.vocab);
        const auto& expected_logits = entry.at("first_logits").array();
        if (expected_prompt.empty() || expected_output.empty() ||
            expected_logits.size() != config.vocab ||
            expected_prompt.size() + expected_output.size() > session.context() ||
            (official && expected_output.size() != 64)) {
            throw std::runtime_error("invalid reference dimensions for " + name);
        }
        shortest = std::min(shortest, expected_prompt.size());
        longest = std::max(longest, expected_prompt.size());

        auto tokens = tokenizer.encode(prompt);
        tokens.insert(tokens.begin(), config.bos);
        if (tokens != expected_prompt || tokenizer.decode(std::span(tokens).subspan(1)) != prompt) {
            throw std::runtime_error("tokenizer byte round-trip or Python token IDs differ for " + name);
        }
        auto logits = session.prefill(tokens);
        double error = 0;
        std::size_t reference_top = 0;
        for (std::size_t i = 0; i < logits.size(); ++i) {
            const auto value = expected_logits[i].number();
            error = std::max(error, std::abs(logits[i] - value));
            if (value > expected_logits[reference_top].number()) {
                reference_top = i;
            }
        }
        if (reference_top != expected_output.front()) {
            throw std::runtime_error("first reference token is not the logit argmax for " + name);
        }
        worst_error = std::max(worst_error, error);
        std::size_t first_mismatch = expected_output.size();
        for (std::size_t step = 0; step < expected_output.size(); ++step) {
            const auto token = greedy.sample(logits);
            if (token == expected_output[step]) {
                ++matched_tokens;
            } else {
                first_mismatch = std::min(first_mismatch, step);
            }
            if (step + 1 < expected_output.size()) {
                logits = session.decode_step(token);
            }
        }
        const bool top_match = first_mismatch != 0;
        const bool full_match = first_mismatch == expected_output.size();
        top_matches += top_match;
        matched_sequences += full_match;
        total_tokens += expected_output.size();
        std::cout << "{\"event\":\"reference_case\",\"id\":\"" << name
                  << "\",\"prompt_tokens\":" << tokens.size()
                  << ",\"max_logit_abs_error\":" << error
                  << ",\"first_mismatch_step\":";
        if (full_match) {
            std::cout << "null";
        } else {
            std::cout << first_mismatch;
        }
        std::cout << "}\n" << std::flush;
    }
    if (official && (cases.size() < 20 || shortest != 16 || longest != 1500)) {
        throw std::runtime_error("Gate 1 needs at least 20 prompts spanning 16 through 1500 tokens");
    }
    const double tolerance = official ? 0.05 : 1e-4;
    const bool passed = matched_sequences == cases.size() && worst_error < tolerance;
    std::cout << "{\"event\":\"" << (official ? "gate_1" : "cpu_smoke")
              << "\",\"passed\":" << (passed ? "true" : "false")
              << ",\"prompts\":" << cases.size()
              << ",\"matched_first_tokens\":" << top_matches
              << ",\"matched_sequences\":" << matched_sequences
              << ",\"matched_tokens\":" << matched_tokens
              << ",\"expected_tokens\":" << total_tokens
              << ",\"max_logit_abs_error\":" << worst_error << "}\n";
    return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string_view mode = argc > 1 ? argv[1] : "";
        if (argc != 4 || (mode != "--gate1" && mode != "--smoke")) {
            throw std::invalid_argument("usage: spark_check (--gate1 | --smoke) MODEL_DIR REFERENCE_JSON");
        }
        return check(argv[2], argv[3], mode == "--gate1");
    } catch (const std::exception& error) {
        std::cerr << "spark_check: " << error.what() << '\n';
        return 1;
    }
}
