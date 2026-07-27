# HTTP server

`mollm_server` is a small, dependency-free (beyond the bundled JSON parser)
OpenAI-compatible HTTP server. It is intended as a local serving baseline, not
as an internet-facing production proxy.

## Build and run

```bash
cmake -G Ninja -B build_i8mm -DCMAKE_BUILD_TYPE=Release
cmake --build build_i8mm -j

./build_i8mm/mollm_server \
  --package qwen35_4b_w4g128.mollm \
  --host 127.0.0.1 --port 8080 --threads 4
```

Metal builds accept `--device metal`. Metal package weights are resident.

## Endpoints

- `GET /v1/models`
- `POST /v1/chat/completions`

Both regular JSON and server-sent event streaming are supported:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mollm",
    "messages": [{"role": "user", "content": "Reply with OK only."}],
    "temperature": 0.8,
    "top_p": 0.95,
    "seed": 42,
    "max_tokens": 32,
    "stream": true
  }'
```

Sampling is configured per request. The server defaults to deterministic
`temperature=0` for backward compatibility, and accepts these fields:

| Field | Range/default | Notes |
|---|---|---|
| `temperature` | 0–2, default 0 | 0 selects greedy decoding |
| `top_p` | 0–1, default 0 | 0 or 1 disables nucleus filtering |
| `seed` | non-negative integer, default 42 | resets the PRNG per request |
| `presence_penalty` | -2–2, default 0 | penalizes tokens present in the penalty window |
| `frequency_penalty` | -2–2, default 0 | scales with occurrence count in the penalty window |
| `top_k` | non-negative integer, default 0 | local extension; 0 disables |
| `min_p` | 0–1, default 0 | local extension; relative to the best token |
| `repeat_penalty` | positive number, default 1 | local extension; 1 disables |
| `repeat_last_n` | -1 or non-negative, default 64 | window shared by all three penalties; 0 disables them and -1 uses the full context |

Authentication, TLS, tool calls, logprobs, parallel requests, and continuous
batching are not implemented. Bind to loopback unless a trusted reverse proxy
provides the missing production controls.

## Cache behavior

The server owns one engine and serializes requests. It retains the engine KV/GDN
state plus the exact token prefix from the previous request. If the next request's
rendered chat prompt extends that prefix, only the suffix is prefetched. If it
diverges, the engine resets before generation. This is the same correctness rule
used by the interactive multi-turn REPL.

Presence, frequency, and repetition penalties use the same consumed-token
history as this KV state. Changing sampling parameters does not invalidate a
valid prefix; changing the rendered messages does.

This single-entry exact-prefix cache is intentionally conservative:

- token equality is required; text equality is not assumed;
- cache length is checked against `engine.past_len()` after every response;
- divergent histories and generation failures invalidate the cache;
- it does not copy KV tensors or retain multiple sessions.

A future multi-session cache should add explicit engine-state snapshot/restore,
byte accounting, capacity limits, and LRU eviction rather than sharing mutable
cache buffers across requests.
