# tests/language_models — AGENTS.md
Purpose: tests of `gpt2_demo_service/language_models/` on a tiny random GPT-2 (no weights needed) plus one opt-in test on the real weights.

## Files
- `test_decoding_cache.py` — in-place writes, views, unchanged storage address on every step, refusal past capacity, position counting.
- `test_cached_forward.py` — cached step-by-step scores equal the full pass to 1e-5, last-position-only output, saved weight names unchanged.
- `test_token_sampling.py` — seeded repeatability, greedy at top-k 1, repetition penalty only on used tokens.
- `test_got_text_generator.py` — end token, stop marker, request stop text, token limit, time limit 0 stops after the first token, prompt-too-long refusal, seeded repeatability.
- `test_real_weights_cache.py` — slow, opt-in (`GOT_RUN_SLOW_TESTS=1`): 20 greedy tokens with the cache equal 20 without it on the real 1.7 GB weights.
