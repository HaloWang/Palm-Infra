#pragma once

#include <string>
#include <utility>
#include <vector>

struct SamplingParams {
    float temperature = 0.6f;       // 0 = greedy
    int top_k = 50;                 // 0 = disabled
    float top_p = 0.9f;             // 0 or 1 = disabled
    float min_p = 0.0f;             // probability relative to the best token
    float repeat_penalty = 1.0f;    // 1 = disabled
    int repeat_last_n = 64;         // window for all penalties; -1 = full context
    float presence_penalty = 0.0f;  // OpenAI-compatible range [-2, 2]
    float frequency_penalty = 0.0f; // OpenAI-compatible range [-2, 2]
    unsigned int seed = 42;
};

bool validate_sampling_params(const SamplingParams& params,
                              std::string* error = nullptr);

// Stateful sampler used by LLMEngine. Its token history follows the tokens
// already consumed by the model, so penalties remain aligned with KV state.
class Sampler {
public:
    explicit Sampler(const SamplingParams& params = {});

    bool configure(const SamplingParams& params, std::string* error = nullptr,
                   bool reset_seed = true);
    void reset();
    void accept(int token_id);
    void accept(const std::vector<int>& token_ids);
    int sample(float* logits, int vocab_size);

    /// Build the exact categorical distribution produced by the configured
    /// temperature/top-k/top-p/min-p filters and repetition penalties.  The
    /// extra history is considered after the committed history without
    /// mutating either the sampler history or RNG state.
    void probabilities(const float* logits, int vocab_size,
                       const std::vector<int>& extra_history,
                       std::vector<float>& output);

    /// Draw from non-negative (not necessarily normalized) weights using the
    /// sampler RNG. Returns zero when the distribution has no finite mass.
    int sample_probabilities(const std::vector<float>& probabilities);

    /// Apply exact speculative rejection sampling for one proposed token.
    /// Returns the proposal with probability min(1, p(x)/q(x)); otherwise
    /// samples from normalized max(0, p-q).
    int speculative_sample(const std::vector<float>& target,
                           const std::vector<float>& proposal,
                           int proposed_token, bool* accepted);

    /// True when sampling is exactly an unmodified argmax over model logits.
    /// Backends may use this to avoid materializing logits on the host.
    bool uses_plain_argmax() const;

    const SamplingParams& params() const { return params_; }
    size_t history_size() const { return history_.size(); }

private:
    SamplingParams params_;
    unsigned int random_state_ = 42;
    std::vector<int> history_;
    std::vector<std::pair<int, float>> saved_logits_;
    std::vector<float> adjusted_logits_;
    std::vector<float> correction_probabilities_;

    float random_uniform();
};

// Backward-compatible stateless entry point.
int sample_token(float* logits, int vocab_size, float temperature, int top_k,
                 float top_p, unsigned int* seed);
