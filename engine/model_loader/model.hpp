#pragma once

#include "engine/tokenizer/tokenizer.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace spark {

struct ModelConfig {
    std::size_t hidden;
    std::size_t intermediate;
    std::size_t layer_count;
    std::size_t heads;
    std::size_t kv_heads;
    std::size_t head_dim;
    std::size_t vocab;
    std::size_t max_positions;
    float rms_epsilon;
    float rope_theta;
    float rope_factor;
    float rope_low_freq;
    float rope_high_freq;
    float rope_original_context;
    TokenId bos;
    std::vector<TokenId> eos;

    [[nodiscard]] std::size_t kv_width() const { return kv_heads * head_dim; }
};

using Weights = std::span<const float>;

struct LayerWeights {
    Weights attention_norm;
    Weights query;
    Weights key;
    Weights value;
    Weights attention_out;
    Weights mlp_norm;
    Weights gate;
    Weights up;
    Weights down;
};

class Model {
public:
    explicit Model(const std::filesystem::path& directory);
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    [[nodiscard]] const ModelConfig& config() const { return config_; }
    [[nodiscard]] const std::string& model_id() const { return model_id_; }
    [[nodiscard]] const std::string& revision() const { return revision_; }
    [[nodiscard]] std::span<const LayerWeights> layers() const { return layers_; }
    [[nodiscard]] Weights embedding() const { return embedding_; }
    [[nodiscard]] Weights final_norm() const { return final_norm_; }
    [[nodiscard]] Weights lm_head() const { return lm_head_; }

private:
    ModelConfig config_{};
    std::string model_id_;
    std::string revision_;
    std::vector<float> storage_;
    std::vector<LayerWeights> layers_;
    Weights embedding_;
    Weights final_norm_;
    Weights lm_head_;
};

}  // namespace spark
