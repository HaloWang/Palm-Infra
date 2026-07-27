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
        float logits[] = {0.5f, 3.0f, 2.0f};
        expect(sampler.sample(logits, 3) == 1, "temperature=0 is greedy");
    }
    {
        SamplingParams p;
        p.temperature = 1.0f;
        p.top_k = 1;
        Sampler sampler(p);
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
    test_validation();
    if (failures != 0)
        return 1;
    std::puts("All sampler tests passed.");
    return 0;
}
