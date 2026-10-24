# language_models — AGENTS.md
Purpose: the two GPT-2 models and everything needed to run them step by step on CPU. Needs torch and tiktoken. Imports no other service package
and no settings module: the answer time limit, weight repo and file names arrive as arguments.

## Files
- `decoding_cache.py` — the per-request key/value cache: allocated once for prompt plus answer, written in place, hands back views of the filled part (never grown); writing past its size raises.
- `gpt_blocks.py` — layer norm, feed-forward, attention (optionally reading and filling a layer cache) and the transformer block; attribute names and the `mask` buffer match the saved weights.
- `gpt_model.py` — the GPT-2 network (small and medium sizes); positions continue after the cached ones, and the output head can run on the last position only.
- `token_sampling.py` — repetition penalty, temperature, top-k and top-p choice of the next token, with an optional seeded random generator.
- `got_text_generator.py` — loads the 1.7 GB Game of Thrones weights and writes an answer for a packed prompt: one read of the prompt, then one token per step; stops at the end token, a stop marker, the token reserve or the time limit, and returns the stop reason and the prefill and decode times.
- `spam_classifier.py` — the spam detector (pads emails to 120 tokens, answers "spam" or "not spam").
- `weights_download.py` — fetches the two weight files from the private weights repo.
