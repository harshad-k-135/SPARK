#include "engine/model_loader/model.hpp"
#include "engine/model_loader/json.hpp"

#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <ranges>
#include <stdexcept>

namespace spark {
namespace {

std::size_t positive_integer(const Json& json, std::string_view key) {
    const auto value = json.at(key).integer();
    if (value == 0) {
        throw std::runtime_error("invalid zero model dimension " + std::string(key));
    }
    return value;
}

float positive_float(const Json& json, std::string_view key) {
    const auto value = static_cast<float>(json.at(key).number());
    if (value <= 0 || !std::isfinite(value)) {
        throw std::runtime_error("invalid model parameter " + std::string(key));
    }
    return value;
}

TokenId special_id(const Json& field, std::size_t vocab) {
    const auto id = field.integer();
    if (id >= vocab || id > std::numeric_limits<TokenId>::max()) {
        throw std::runtime_error("model special token is outside the vocabulary");
    }
    return static_cast<TokenId>(id);
}

ModelConfig parse_config(const Json& config) {
    if (config.at("model_type").string() != "llama" ||
        config.at("hidden_act").string() != "silu") {
        throw std::runtime_error("expected a Llama decoder with SwiGLU");
    }
    for (const auto key : {"attention_bias", "mlp_bias"}) {
        if (const auto* setting = config.find(key); setting && setting->boolean()) {
            throw std::runtime_error("SPARK v1 requires a bias-free Llama decoder");
        }
    }
    const auto& rope = config.at("rope_scaling");
    if (rope.at("rope_type").string() != "llama3") {
        throw std::runtime_error("expected Llama 3 RoPE scaling");
    }
    ModelConfig result{
        .hidden = positive_integer(config, "hidden_size"),
        .intermediate = positive_integer(config, "intermediate_size"),
        .layer_count = positive_integer(config, "num_hidden_layers"),
        .heads = positive_integer(config, "num_attention_heads"),
        .kv_heads = positive_integer(config, "num_key_value_heads"),
        .head_dim = positive_integer(config, "head_dim"),
        .vocab = positive_integer(config, "vocab_size"),
        .max_positions = positive_integer(config, "max_position_embeddings"),
        .rms_epsilon = positive_float(config, "rms_norm_eps"),
        .rope_theta = positive_float(config, "rope_theta"),
        .rope_factor = positive_float(rope, "factor"),
        .rope_low_freq = positive_float(rope, "low_freq_factor"),
        .rope_high_freq = positive_float(rope, "high_freq_factor"),
        .rope_original_context = positive_float(rope, "original_max_position_embeddings"),
        .bos = 0,
        .eos = {},
    };
    if (result.hidden % result.heads != 0 ||
        result.hidden / result.heads != result.head_dim || result.head_dim % 2 != 0 ||
        result.heads % result.kv_heads != 0 || result.rope_high_freq <= result.rope_low_freq) {
        throw std::runtime_error("invalid attention dimensions or RoPE scale in model config");
    }
    result.bos = special_id(config.at("bos_token_id"), result.vocab);
    const auto& eos = config.at("eos_token_id");
    if (eos.is_array()) {
        for (const auto& token : eos.array()) {
            result.eos.push_back(special_id(token, result.vocab));
        }
    } else {
        result.eos.push_back(special_id(eos, result.vocab));
    }
    if (result.eos.empty()) {
        throw std::runtime_error("model config lacks an end-of-sequence token");
    }
    return result;
}

Weights tensor(const Json& manifest, Weights storage, std::string_view name,
               std::initializer_list<std::size_t> dimensions) {
    const auto& entry = manifest.at(name);
    const auto& shape = entry.at("shape").array();
    if (shape.size() != dimensions.size() || entry.at("dtype").string() != "F32") {
        throw std::runtime_error("unexpected tensor rank or dtype for " + std::string(name));
    }
    auto axis = shape.begin();
    std::size_t count = 1;
    for (auto dimension : dimensions) {
        if ((axis++)->integer() != dimension || dimension > storage.size() / count) {
            throw std::runtime_error("invalid weight shape for " + std::string(name));
        }
        count *= dimension;
    }
    const auto offset = entry.at("offset").integer();
    if (offset % sizeof(float) != 0 || offset / sizeof(float) > storage.size() - count) {
        throw std::runtime_error("out-of-range tensor offset for " + std::string(name));
    }
    return storage.subspan(offset / sizeof(float), count);
}

}  // namespace

Model::Model(const std::filesystem::path& directory) {
    static_assert(std::endian::native == std::endian::little);
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    const auto manifest = Json::read(directory / "manifest.json");
    if (manifest.at("format").string() != "spark-f32-v1") {
        throw std::runtime_error("expected a spark-f32-v1 model bundle");
    }
    model_id_ = manifest.at("model_id").string();
    revision_ = manifest.at("revision").string();
    config_ = parse_config(manifest.at("config"));

    const auto weights_path = directory / "weights.bin";
    const auto bytes = std::filesystem::file_size(weights_path);
    if (bytes == 0 || bytes % sizeof(float) != 0 || bytes / sizeof(float) > storage_.max_size()) {
        throw std::runtime_error("invalid weights.bin size");
    }
    storage_.resize(static_cast<std::size_t>(bytes / sizeof(float)));
    std::ifstream weights(weights_path, std::ios::binary);
    constexpr std::size_t block = 1U << 22;
    for (std::size_t offset = 0; offset < storage_.size(); offset += block) {
        const auto length = std::ranges::min(block, storage_.size() - offset);
        if (!weights.read(reinterpret_cast<char*>(storage_.data() + offset),
                          static_cast<std::streamsize>(length * sizeof(float)))) {
            throw std::runtime_error("could not read complete weights.bin");
        }
    }

    const auto& entries = manifest.at("tensors");
    const auto h = config_.hidden;
    const auto kv = config_.kv_width();
    const auto m = config_.intermediate;
    embedding_ = tensor(entries, storage_, "model.embed_tokens.weight", {config_.vocab, h});
    final_norm_ = tensor(entries, storage_, "model.norm.weight", {h});
    lm_head_ = tensor(entries, storage_, "lm_head.weight", {config_.vocab, h});
    for (std::size_t layer = 0; layer < config_.layer_count; ++layer) {
        const auto prefix = "model.layers." + std::to_string(layer) + ".";
        const auto get = [&](std::string_view name, std::initializer_list<std::size_t> dims) {
            return tensor(entries, storage_, prefix + std::string(name), dims);
        };
        layers_.push_back({
            .attention_norm = get("input_layernorm.weight", {h}),
            .query = get("self_attn.q_proj.weight", {h, h}),
            .key = get("self_attn.k_proj.weight", {kv, h}),
            .value = get("self_attn.v_proj.weight", {kv, h}),
            .attention_out = get("self_attn.o_proj.weight", {h, h}),
            .mlp_norm = get("post_attention_layernorm.weight", {h}),
            .gate = get("mlp.gate_proj.weight", {m, h}),
            .up = get("mlp.up_proj.weight", {m, h}),
            .down = get("mlp.down_proj.weight", {h, m}),
        });
    }
}

}  // namespace spark
