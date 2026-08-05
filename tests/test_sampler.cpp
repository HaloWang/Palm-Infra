#include "engine/sampler.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

void test_greedy_and_filters() {
    {
        SamplingParams p;
        p.temperature = 0.0f;
        Sampler sampler(p);
        expect(sampler.uses_plain_argmax(),
               "temperature=0 exposes the plain-argmax fast path");
        float logits[] = {0.5f, 3.0f, 2.0f};
        expect(sampler.sample(logits, 3) == 1, "temperature=0 is greedy");
    }
    {
        SamplingParams p;
        p.temperature = 1.0f;
        p.top_k = 1;
        Sampler sampler(p);
        expect(sampler.uses_plain_argmax(),
               "top_k=1 exposes the plain-argmax fast path");
        float logits[] = {0.5f, 3.0f, 2.0f};
        expect(sampler.sample(logits, 3) == 1, "top_k=1 is greedy");
    }
    {
        SamplingParams p;
        p.temperature = 1.0f;
        p.top_k = 0;
        p.top_p = 0.0f;
        p.min_p = 0.5f;
        Sampler sampler(p);
        float logits[] = {4.0f, 0.0f, -1.0f};
        expect(sampler.sample(logits, 3) == 0,
               "min_p removes tokens far below the best probability");
    }
    {
        SamplingParams p;
        p.temperature = 1.0f;
        p.top_k = 0;
        p.top_p = 0.5f;
        Sampler sampler(p);
        float logits[] = {4.0f, 0.0f, -1.0f};
        expect(sampler.sample(logits, 3) == 0,
               "top_p retains the minimum prefix reaching the cutoff");
    }
}

void test_penalties_and_logit_restore() {
    {
        SamplingParams p;
        p.temperature = 0.0f;
        p.presence_penalty = 1.0f;
        Sampler sampler(p);
        expect(!sampler.uses_plain_argmax(),
               "penalties disable the plain-argmax fast path");
        sampler.accept(0);
        float logits[] = {3.0f, 2.5f};
        expect(sampler.sample(logits, 2) == 1,
               "presence penalty changes greedy selection");
        expect(logits[0] == 3.0f && logits[1] == 2.5f,
               "presence penalty restores caller logits");
    }
    {
        SamplingParams p;
        p.temperature = 0.0f;
        p.frequency_penalty = 0.5f;
        Sampler sampler(p);
        sampler.accept(0);
        sampler.accept(0);
        float logits[] = {3.0f, 2.5f};
        expect(sampler.sample(logits, 2) == 1,
               "frequency penalty scales with token count");
    }
    {
        SamplingParams p;
        p.temperature = 0.0f;
        p.repeat_penalty = 2.0f;
        p.repeat_last_n = 8;
        Sampler sampler(p);
        sampler.accept(0);
        float logits[] = {3.0f, 2.5f};
        expect(sampler.sample(logits, 2) == 1,
               "repeat penalty applies inside the recent window");
    }
    {
        SamplingParams p;
        p.temperature = 0.0f;
        p.presence_penalty = 1.0f;
        p.repeat_last_n = 1;
        Sampler sampler(p);
        sampler.accept(0);
        sampler.accept(1);
        float logits[] = {3.0f, 1.0f, 2.5f};
        expect(sampler.sample(logits, 3) == 0,
               "presence penalty ignores tokens outside repeat_last_n");
    }
    {
        SamplingParams p;
        p.temperature = 0.0f;
        p.frequency_penalty = 1.0f;
        p.repeat_last_n = 2;
        Sampler sampler(p);
        sampler.accept(0);
        sampler.accept(0);
        sampler.accept(1);
        float logits[] = {4.0f, 1.0f, 2.5f};
        expect(sampler.sample(logits, 3) == 0,
               "frequency counts only tokens inside repeat_last_n");
    }
    {
        SamplingParams p;
        p.temperature = 0.0f;
        p.presence_penalty = 2.0f;
        p.frequency_penalty = 2.0f;
        p.repeat_penalty = 2.0f;
        p.repeat_last_n = 0;
        Sampler sampler(p);
        sampler.accept(0);
        float logits[] = {3.0f, 2.5f};
        expect(sampler.sample(logits, 2) == 0,
               "repeat_last_n=0 disables all repetition penalties");
    }
}

void test_seed_and_reset() {
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_k = 3;
    p.top_p = 1.0f;
    p.seed = 1234;
    Sampler a(p), b(p);
    for (int i = 0; i < 32; ++i) {
        float logits_a[] = {1.0f, 0.9f, 0.8f};
        float logits_b[] = {1.0f, 0.9f, 0.8f};
        const int ta = a.sample(logits_a, 3);
        const int tb = b.sample(logits_b, 3);
        expect(ta == tb, "same seed produces the same sample stream");
        a.accept(ta);
        b.accept(tb);
    }
    a.reset();
    b.reset();
    expect(a.history_size() == 0 && b.history_size() == 0,
           "reset clears sampling history");
    float logits_a[] = {1.0f, 0.9f, 0.8f};
    float logits_b[] = {1.0f, 0.9f, 0.8f};
    expect(a.sample(logits_a, 3) == b.sample(logits_b, 3),
           "reset restores the configured seed");
}

void test_probability_and_rejection_sampling() {
    {
        SamplingParams p;
        p.temperature = 1.0f;
        p.top_k = 0;
        p.top_p = 1.0f;
        Sampler sampler(p);
        const float logits[] = {0.0f, std::log(2.0f), std::log(3.0f)};
        std::vector<float> probabilities;
        sampler.probabilities(logits, 3, {}, probabilities);
        expect(probabilities.size() == 3,
               "probability builder returns the complete vocabulary");
        expect(std::fabs(probabilities[0] - 1.0f / 6.0f) < 1e-5f &&
                   std::fabs(probabilities[1] - 2.0f / 6.0f) < 1e-5f &&
                   std::fabs(probabilities[2] - 3.0f / 6.0f) < 1e-5f,
               "probability builder matches temperature softmax");
    }
    {
        SamplingParams p;
        p.temperature = 0.0f;
        p.presence_penalty = 2.0f;
        Sampler sampler(p);
        const float logits[] = {3.0f, 2.0f};
        std::vector<float> probabilities;
        sampler.probabilities(logits, 2, {0}, probabilities);
        expect(probabilities[0] == 0.0f && probabilities[1] == 1.0f,
               "uncommitted proposal history participates in penalties");
        expect(sampler.history_size() == 0,
               "probability construction does not commit proposal history");
    }
    {
        SamplingParams p;
        p.temperature = 1.0f;
        p.top_k = 0;
        p.top_p = 1.0f;
        p.seed = 1234;
        Sampler sampler(p);
        const std::vector<float> target = {0.1f, 0.3f, 0.6f};
        const std::vector<float> proposal = {0.5f, 0.4f, 0.1f};
        int counts[3] = {0, 0, 0};
        constexpr int trials = 50000;
        for (int i = 0; i < trials; ++i) {
            const int proposed = sampler.sample_probabilities(proposal);
            bool accepted = false;
            const int sampled = sampler.speculative_sample(
                target, proposal, proposed, &accepted);
            (void)accepted;
            ++counts[sampled];
        }
        for (int token = 0; token < 3; ++token) {
            const float observed =
                static_cast<float>(counts[token]) / trials;
            expect(std::fabs(observed - target[token]) < 0.01f,
                   "p/q rejection sampling reproduces the target distribution");
        }
    }
}

void test_validation() {
    SamplingParams p;
    std::string error;
    p.temperature = 2.1f;
    expect(!validate_sampling_params(p, &error) && !error.empty(),
           "temperature range is validated");
    p = SamplingParams{};
    p.repeat_last_n = -2;
    expect(!validate_sampling_params(p, &error),
           "repeat_last_n range is validated");
    p = SamplingParams{};
    p.frequency_penalty = -2.0f;
    p.presence_penalty = 2.0f;
    expect(validate_sampling_params(p, &error),
           "OpenAI penalty boundary values are accepted");
}

} // namespace

int main() {
    test_greedy_and_filters();
    test_penalties_and_logit_restore();
    test_seed_and_reset();
    test_probability_and_rejection_sampling();
    test_validation();
    if (failures != 0)
        return 1;
    std::puts("All sampler tests passed.");
    return 0;
}
