# Retrieval for the Game of Thrones GPT-2: design (2026-10-24)

Workflow `wf-1024-2000-got-rag`, part 4 (architect), revised in part 6 after the senior review of part 5 (§16, §17).
DECISIONS key `hq.gpt2_got_rag_1024` (Karam's go, deploy to the Space included).
Inputs: Karam's own RAG study notes and his runnable demo (part 1), the live-system inventory `../rag-inventory-1024.md` (part 2),
the web research (part 3). Every builder reads this file first, then the AGENTS.md of each folder it touches.

**Frozen code that already exists:** `../gpt2_service_huggingface_space/retrieval_contracts/` (the contract, §6, including the
shared answer locator `rare_shingle_location.py`, §6.6) and its `shared_fixture/` (the shared test fixture, §14.4). 11 self-tests
pass (`python -m pytest retrieval_contracts` from the service folder). Builders import them and never edit them.

---

## 0. One-paragraph summary

The model sounds like the books but guesses facts. We add retrieval in front of it: a small offline-built index over the book text
(about 12k passages of about 200 GPT-2 tokens, never crossing a chapter) and over the 34,001 cleaned Q&A pairs. Per question the
Space runs lexical (own BM25) and dense (bge-small, ONNX on CPU) search, fuses them with RRF (k = 60), optionally reranks the top
10 with a small cross-encoder, gates on one calibrated score, packs at most 3 passages and 2 pair examples into the model's EXACT
trained template within a hard 1,024-token budget (the default 600 retrieved tokens hold 2 examples + 2 passages, or 3 passages
alone), generates with a new preallocated KV cache (without it a 600-token prompt costs about 3.3 s per token) under a 25 s answer
time limit, cuts any long verbatim copy, and returns `{"completion"}` exactly as today plus `retrieval_used` and `sources`
(book + chapter for each source, and ONE book snippet of at most 25 words per answer). The index lives only in a PRIVATE Hugging
Face dataset repo, pinned by commit, verified by checksums at startup. Research says a 355M model may ignore pasted context, so the
evaluation decides between four prompt-only layouts on a fact-dev set it never reports as a result, and, by a fixed rule, whether a
RAFT grounding fine-tune on free Kaggle GPU follows.

## 1. Goals, non-goals, acceptance targets

### 1.1 Goals
1. Fewer invented facts on the public Space (the "Gregor Clegane sits on the Iron Throne" class of error).
2. Public answers name their evidence: book + chapter for each source, and one snippet of at most 25 words per answer.
3. Senior-grade structure: small folders with one job each, typed interfaces, every default value in one settings module, tests, no dead code.
4. Nothing breaks: the portfolio contract, the spam demo, the limits, the rollback.

### 1.2 Non-goals
- No new hardware, no paid API at runtime, no LLM at runtime other than the GPT-2 itself.
- No change to the training template text (the "qusetion" typo stays byte-identical).
- No direct return of stored pair answers as the completion (decision D9).
- No portfolio change here (a later part does it on a branch; the API only gains optional fields).

### 1.3 Acceptance targets

"Today" = the live mixed model measured on Kaggle (`kaggle_finetune/mixed_run_results/comparison.json`) or on the live Space
(inventory §5). Because sampling differs by machine, the evaluation harness re-measures the no-retrieval baseline itself with
the same seeds, and every gate compares like with like (harness baseline vs harness retrieval run). A1-A4 are measured ONCE, on the
baseline and on the configuration already chosen on the dev sets (§11.6); nothing is tuned on them.

| # | Measure (how) | Today | Ship gate | Target |
|---|---|---|---|---|
| A1 | 30 fact questions, right in 2 of 3 seeded samples (`fact_questions.is_right`) | 10/30 (35/90 samples) | ≥ 15/30 and at most 2 of the 10 previously right turn wrong | ≥ 20/30 |
| A2 | Same, without the 3 questions that are word for word in the pairs | 8/27 | ≥ 12/27 | ≥ 17/27 |
| A3 | The 10 known questions (first 10 of the 30) | 4/10 (13/30 samples) | ≥ 6/10 | ≥ 8/10 |
| A4 | Regression set (Iron Throne x2, Kingslayer, Theon's home, §11.1) | all wrong | Iron Throne answers never name Gregor / Clegane / the Mountain | all right |
| A5 | Passage hit@5 on the held-out test blocks (§11.1 G1, about 196 questions), shipped retrieval settings | none | ≥ 0.70 | ≥ 0.80 |
| A6 | Mean reciprocal rank at 10 on the same questions | none | reported | ≥ 0.55 |
| A7 | Evidence gate: off-topic / not-in-books questions that still get evidence (G3) | n/a | ≤ 15 % while ≥ 85 % of G1 dev questions get evidence | ≤ 10 % / ≥ 90 % |
| A7b | Fact-dev questions (FD, §11.1) that get evidence at the chosen floor | n/a | ≥ 70 % | ≥ 80 % |
| A8 | Pair-example precision at the chosen threshold (leave-one-out held-back half, §11.4) | n/a | ≥ 0.90 | ≥ 0.95 |
| A9 | Retrieval stages p95 on the Space (Server-Timing header, 20 warm calls) | n/a | ≤ 400 ms reranker off, ≤ 800 ms on | ≤ 250 / ≤ 600 ms |
| A10 | GoT `/generate` p95, warm, one caller, 20 calls, retrieval on; the line cap then obeys (cap − 1) × p95 + p95 ≤ 60 s (§10.3) | 6.7-14.5 s (3 calls, no retrieval) | ≤ 22 s | ≤ 16 s |
| A11 | Plain-path seconds per generated token (KV cache) | 0.5-1.05 s | ≤ 0.45 s | ≤ 0.35 s |
| A12 | Spam `/generate` p95 | < 1 s | ≤ 1.5 s, body still exactly `{"completion"}` | unchanged |
| A13 | Memory steady / peak (`/health`) | 2.7 / 3.9 GB | ≤ 4.5 / ≤ 6 GB | ≤ 3.8 / ≤ 5 GB |
| A14 | Restart to ready | about 31 s | ≤ 120 s | ≤ 75 s |
| A15 | Contract: the old portfolio body returns 200 with a string `completion`; no `text` key ever; GoT body keys exactly {completion, retrieval_used, sources}, each source exactly {source_type, book, chapter, snippet} | holds | holds | holds |
| A16 | Copyright: no response field shares more than 25 consecutive words with ANY passage of the bundle (offline, not only the packed ones); at most one book snippet per answer, ≤ 25 words | n/a | 0 violations over every evaluation answer | 0 |

The answer-quality numbers on the 1,233 held-out questions (token F1, key-term recall, unsupported-name rate) are reported, with
one gate only: retrieval must not be worse than the baseline on key-term recall.

## 2. Facts the design stands on (from the inventory; numbers there)

1. No KV cache today: each new token re-runs the whole sequence (about 0.34 s + 0.0049 s per context token). RAG needs incremental decoding.
2. Context 1,024 tokens with learned positions; today the code silently keeps the LAST 1,024 tokens, which would cut the header off.
3. The model never saw a context slot: trained on header + question + answer (at most 142 tokens) and, separately, plain book blocks.
   Positions: in the fine-tune data we can inspect (the mixed run), the template appeared only at positions 0-142 and book text only
   at positions 0-511 (512-token blocks); only OpenAI's pretraining used all 1,024 positions. Every retrieval layout therefore moves
   the question block to positions it was never fine-tuned at, and RAFT sequences (§12) must reach about 1,000 tokens.
4. Exact template (inventory §2), including "qusetion"; prompt masked in both Kaggle runs; examples padded with token 50256, one example per sequence.
5. Question cap 500 characters; retrieved text is built server-side, never sent in `prompt`.
6. Portfolio reads `data.text || data.completion`: never add `text`; `completion` stays the plain answer.
7. Dockerfile copies only top-level files and the publish script uploads only `*.py *.html requirements.txt Dockerfile README.md`.
8. No persistent disk on the Space; weights and index come from private repos at every start with the `HF_TOKEN` secret.
9. Index sources: `question_answer_train.json` (34,001 pairs, 54 % located to ±20k characters) and `book_train_parts.json` (6 parts),
   plus `books_validation_text.txt` (the held-out Feast stretch, 261,252 characters).
10. No chapter data exists; the raw books in `~/Downloads/drive-download-20250818T222213Z-1-001/` keep heading lines and contents lists.
11. 3 of the 30 fact questions are word for word in the pairs (Arya's sword, Lannister sigil, Baratheon sigil).
12. Memory headroom about 12 GB; one generation at a time; 4 in line; 20 per minute per address. The reference browser client
    (`gpt2_space_client_example.mjs`, `REQUEST_TIMEOUT_MILLISECONDS = 60000`) gives up after 60 s, queue wait included.
13. The live model's starting point (Karam's 2025 original) trained on part 6's book text and on the 325 training pairs that share
    that passage (both runs of 2026 and the index leave those pairs out; nothing ever trained on the 1,233 held-out questions). G1
    answer scores (§11.1) are therefore inflated in BOTH arms, baseline and retrieval; the report must say so. Retrieval metrics are
    not affected.
14. Measured with the frozen locator (`rare_shingle_location.py`, probe 2026-10-24, 5 s to index, about 1 GB): of the 816 answerable
    held-out questions, 443 are located in part 6 (median 4 rare hits, 64 characters); 58 sit within 2,000 characters of a block edge
    of §11.1, leaving 189 dev and 196 test. Of the 34,001 pairs, 17,488 are placed by shingles inside their coarse window, 875 have a
    window but no shingle cluster, 2,221 of the 15,638 unlocated pairs are placed corpus-wide, 13,417 stay unplaced. On 2,001 sampled
    pairs the shingle place agreed with the coarse location 1,015 times and disagreed 17 times (98 %); a one-hit rule fell to 96 %.

## 3. How this maps to Karam's own five stages

| His stage | What we keep | What we change for this project, and why |
|---|---|---|
| Ingest: chunk at structure, small overlap, size by testing | sentence-aligned chunks inside one chapter, 1-sentence overlap, size chosen by ablation (128/200/256) | the 1,024-token window, not the embedder, sets the size |
| Index: same embedding model for chunks and queries, store text next to vectors, BM25 too | one pinned encoder, one code path offline and online, records stored beside the vectors | a flat exact search instead of a vector DB: 50k x 384 floats is about 70 MB |
| Retrieve: BM25 + vector, RRF k = 60, cross-encoder rerank, retrieve wide rerank narrow | dense 50 + lexical 50 → fused 20 → rerank the top 10 → 3 | the reranker must earn its place on 2 vCPU (switch, decided by ablation; cost prior in D8) |
| Answer: chunks are data, not orders; say when it is not in the books; cite | passages never carry instructions; citations in a separate field; score-gated evidence | GPT-2 cannot follow "use only these" or loop, so the gate and the guard are code |
| Evaluate / operate: golden set by file + section, recall@k, precision@k, MRR, nDCG, faithfulness, p95, no-answer rate, rerun | gold stored as rare-shingle hits (character spans), all metrics plus ablations and per-stage p95, per-request log, index version in /health | the golden set comes free from the held-out pairs (split by contiguous blocks, not alternate rows); tuning sets and reported sets never mix; no LLM judge at runtime |

His rule that fits our biggest risk: when the right passage was retrieved but the answer is still wrong, the fix is in the
prompt or the model, not in retrieval settings. That is exactly the RAFT branch (D22).

## 4. Decisions (decision, reason, rejected options)

**D1. Two evidence corpora in one bundle, retrieved separately, merged only when packing.** Passages and pairs have different
shapes, score scales and prompt forms. Rejected: one mixed ranked list (pair cosines and passage cosines are not comparable);
pairs only (46 % unlocated, Storm has 3.8 % of the pairs for 24 % of the text, answers vague); books only (pairs are the model's
trained answer style and catch paraphrased questions).

**D2. Sentence-aligned passages of about 200 GPT-2 tokens (hard ceiling 230), 1-sentence overlap, never crossing a chapter or a
part; final size by ablation (target / ceiling 128 / 150, 200 / 230, 256 / 290) on the G1 dev blocks.** The default 600-token
budget holds 2 pair examples (about 150 tokens) + 2 passages, or 3 passages without examples; the research shows small chunks keep
recall and precision. Rejected: fixed character windows (cut sentences), 512+ token chunks
(only one would fit, vectors get diluted), parent-child retrieval (no room for parents in 1,024 tokens), LLM semantic chunking
(cost, no evidence of gain on fiction).

**D3. A fixed heading in front of every indexed passage** ("A Storm of Swords, Tyrion (chapter 12): ..."), used by the vector,
the lexical postings and the reranker; the prompt and the citation use the raw text. This is the no-model form of contextual
retrieval (point-of-view chapters make the heading informative). Rejected: an LLM-written context sentence per chunk (needs an API
and a paid offline run for a small expected gain here).

**D4. Dense encoder BAAI/bge-small-en-v1.5 at commit `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, its own `onnx/model.onnx`,
run with `onnxruntime` + `tokenizers`, CLS pooling, unit vectors; the SAME class encodes offline and online.** Best retrieval
score of the CPU-sized models (MTEB retrieval 51.7), 384 dimensions, about 10-30 ms per question. Passage search prefixes the
question with the bge instruction; question-to-question matching uses no prefix. Rejected: all-MiniLM-L6-v2 (10 points worse
retrieval), e5-small-v2 (needs prefixes, slightly worse), gte-small (no gain), bge-base (3x compute for little gain at this scale),
API embeddings (outbound dependency, cost), sentence-transformers or transformers at runtime (heavy image, version coupling with
the pinned `huggingface_hub 2.3.0`), int8 quantisation (accuracy risk, speed not needed).

**D5. Exact inner-product search in numpy over float32 unit vectors.** About 12k + 34k vectors x 384 x 4 B ≈ 70 MB, a few
milliseconds per query, zero approximation error. Rejected: HNSW / IVF / FAISS (they pay off at about a million vectors and add
build parameters and misses), Chroma / Qdrant / Pinecone / pgvector (a server or a managed service for a static, read-only corpus;
a managed service would also put copyrighted text outside the private repo).

**D6. Lexical search is our own BM25 module** (Lucene idf `ln(1 + (N - df + 0.5)/(df + 0.5))`, k1 = 1.2, b = 0.75, document
length in lexical tokens), with weights precomputed per (term, document) at build time and summed at query time with
`numpy.bincount` over distinct query terms. Names like Clegane, Harrenhal and Needle are exactly what dense small models miss.
`rank_bm25` is the ordering comparison in the builder's tests. Rejected: `bm25s` (fine, but adds scipy for something we write in
60 lines and Karam planned to write it himself), `rank_bm25` at runtime (about 4.5 queries per second), Elasticsearch (a server).

**D7. Hybrid = dense top 50 + lexical top 50, fused by RRF (k = 60, 1-based ranks, ties by id), keep the fused top 20.** Hybrid
is the default and ships unless a single search BEATS it by ≥ 2 points of G1 dev hit@5 (`passage_search_mode`). The G1 questions
were written from 200-character chunks and reuse the passage's own words, so that set favours lexical search; visitor questions are
general, and hybrid is the robust choice when the two are close. Rejected: weighted score sums (need fragile score normalisation),
learned fusion (no training data), "simpler search when merely within 2 points" (biased by the lexical-heavy dev set).

**D8. Reranker = cross-encoder ms-marco-MiniLM-L6-v2 at commit `233902d25c440f23af6f7d6e94d2946bac0bee0a` (ONNX), over the top
`rerank_candidate_count` = 10 of the fused 20, each input capped at 256 word pieces, behind `reranker_enabled`.** Cost prior: the
model needs about 22 MFLOP per word piece, so 20 candidates x about 260 pieces is about 110 GFLOP, roughly 1-2 s on 2 vCPU (already
over A9's 800 ms); 10 x ≤ 256 is about 55 GFLOP, roughly 0.4-1 s. Expect it to stay off unless the gain is large. On only if it
adds ≥ 3 points hit@3 on G1 dev AND keeps A9 (the ablation's "Mac time x 3" estimate is checked against this prior, then live).
Rejected: bge-reranker-base (278M parameters, 1-2 GB, seconds on 2 vCPU), LLM reranking (no LLM besides GPT-2).

**D9. Pair matching = question-to-question cosine (symmetric, no prefix) fused by RRF with lexical search over question + answer;
pairs above one threshold, calibrated leave-one-out on the training pairs (§11.4), become in-format EXAMPLES in the prompt, never
the completion.** The demo's answer must be
the model's; stored answers are another model's text with interpretive tails. A near-identical example right before the question
is the strongest signal a 355M model can copy. Rejected: direct return of a matched answer (misrepresents the demo), a semantic
cache (same objection), two thresholds (only one action exists).

**D10. One evidence gate per question:** the best dense cosine among the passage candidates (or the best rerank score when the
reranker is on) must reach a calibrated floor (chosen on G1 dev, checked on the general fact-dev questions, A7b); below it, no
passages. With no passages and no pair above threshold the answer
uses today's exact prompt and says `retrieval_used: false`. Rejected: an LLM "is this enough?" check, answer loops and query
rewriting (GPT-2 cannot do them), refusing by default (the demo should still answer in the voice of the books; the no-evidence
rate is measured and can be switched to refusal later if a RAFT model learns abstention).

**D11. The prompt keeps the trained template byte for byte; where retrieved text sits is chosen by evaluation among four
prompt-only layouts (§7.3); the best item sits nearest the question.** Rejected: an untrained "### context:" slot without
fine-tuning (small models ignore or garble it), editing the template (breaks what the model learned).

**D12. Hard token budget:** prompt tokens + answer reserve (max(request tokens, 120), at most 200) + 8 safety ≤ 1,024, and
retrieved text ≤ `context_token_budget` (600 by default; ablation 300 / 450 / 600). The question is never cut; items are taken
best first whole, and only the last item taken is trimmed (at a sentence end) to fit; the lowest-ranked item is dropped until the
rendered prompt fits. Passage length is the bundle's chunking ceiling, not a prompt setting. The silent left-truncation is removed.

**D13. Incremental decoding (KV cache, preallocated per request to prompt + max_new_tokens positions and written in place) and an
output head on the last position only.** Required for any long prompt (inventory fact 1) and it speeds up today's plain answers as
well. A cache grown by `torch.cat` would copy about 150 MB per token at 750 positions (24 layers x keys and values), so it is never
grown. Weight names and the saved `mask` buffer stay identical. Rejected: short contexts only (kills retrieval), int8 / bf16 weights
(changes outputs; possible later step), paid hardware (Karam's call).

**D14. The bundle is built offline on the Mac, stored ONLY in the private dataset repo `karamAM/gpt2-got-retrieval-index`, pinned
by full commit hash, downloaded at Space start with `HF_TOKEN`, verified (manifest checksums, format and tokenizer versions,
encoder fingerprint) and held read-only in memory.** Rejected: baking it into the image (public Space images can be pulled),
committing it to the Space repo (public), Space persistent storage (gone / paid), building at startup (slow, needs book text in
the image).

**D15. The two public encoder models are baked into the image at build time from their pinned commits** (MIT and Apache-2.0
licences); at startup the service hashes its ONNX file and refuses retrieval if the fingerprint differs from the manifest.

**D16. One index for evaluation and production: books = the 6 training parts + the held-out passage (part 6); pairs = the 34,001
training pairs only (the 1,233 held-out pairs are never indexed).** This overrides the inventory's two-variant default on
purpose: (a) the held-out questions can only test retrieval if their passage is retrievable; (b) the leak that matters is
pair-to-pair (a matcher answering by lookup), and the held-out pairs stay out; (c) we evaluate exactly what we ship, through the
same answer path (`GroundedAnswerService.answer_from_evidence`, §7.6). RAFT data (§12) never uses part-6 passages, so the held-out
questions stay a clean test of the trained model. (d) The 2025 starting model did read part 6 (§2 fact 13): G1 answer scores are
inflated in both arms and the report says so.

**D17. API:** `completion` unchanged; Game of Thrones answers gain `retrieval_used` and `sources[{source_type, book, chapter,
snippet}]`; spam body stays exactly `{"completion"}`; never a `text` field; request gains optional `use_retrieval` (default true);
a `Server-Timing` header carries stage times. Two explicit response models, not one model with `response_model_exclude_none`
(that option is recursive: it would also drop `chapter: null` inside a source, so source objects would come back with different
keys; tested on fastapi 0.111.0 + pydantic 2.11.7): `CompletionResponse{completion}` for spam and
`GroundedCompletionResponse{completion, retrieval_used, sources}` for Game of Thrones, where every `SourcePayload` always carries
all four keys (`book`, `chapter`, `snippet` may be null). Citations do not go inside `completion` (the portfolio's cleanup code
expects the plain answer).

**D18. Copyright guard in code:** Karam's rule is "book + chapter and at most a 25-word snippet", read the safe way: ONE book-text
snippet per response, from the top included passage, made of its first ≤ 25 words (passages start at a sentence start), so the
snippet does not depend on the question and a caller cannot steer it to an arbitrary stretch of the books. Other passage sources
show book + chapter with `snippet: null`; a pair source's snippet is its question (written by gpt-4o-mini, not book text, also
≤ 25 words). A switch `book_snippet_enabled` (env `GOT_BOOK_SNIPPET_ENABLED`) turns book snippets off without a deploy. A runtime
verbatim guard cuts the completion if it shares more than 25 consecutive words with any PACKED passage; the model was trained on the
books and can also repeat memorised text, so A16 is measured offline against ALL passages. If A16 finds a run outside the packed
passages, a corpus-wide guard follows: a new bundle file of 26-word shingle hashes over every passage (about 2M uint64, about
16 MB), bundle format version 2. No book text or visitor text in logs (Space logs must be treated as public); the README states the
policy. Rejected: trusting the model, a judge model, a question-steered snippet window (aimable at any 25 words), one snippet per
source (up to three book snippets per answer).

**D19. Retrieval never breaks `/generate`:** a bundle failure at startup turns retrieval off and `/health` says why; an error in
one request falls back to the plain prompt and is logged. Only model errors still return 500.

**D20. Configuration in one place per package:** `gpt2_demo_service/service_settings.py` holds EVERY default value of the service
and is the only reader of the environment (a handful of `GOT_*` overrides plus the existing `TORCH_THREADS`, `WEIGHTS_REPO_ID`,
`HF_TOKEN`); inside the service only `web_api/app_factory.py` imports it, and the evaluation imports it to start from the shipped
values. Each runtime package declares the frozen dataclass it consumes, WITHOUT defaults, next to its code
(`index_search/evidence_search_settings.py`, `grounded_answers/answer_settings.py`, `web_api/request_limits.py`), so
`index_search` never pulls in pydantic or torch and there is no import cycle. Settings are plain frozen dataclasses; pydantic stays
for the API schemas only. Offline: `retrieval_index_builder/index_build_settings.py`, `retrieval_evaluation/evaluation_settings.py`.
Calibrated values (thresholds, layout, switches, bundle revision) are written as code defaults from the evaluation's
`recommended_settings.json`; never tuned by hand on the live Space.

**D21. No LangChain, LangGraph, Chroma, file watcher, Ollama, access labels or runtime LLM judge.** A 2 vCPU image and a static,
single-audience corpus do not need them; GPT-2 cannot drive them. (LangChain names map 1:1 to our modules if a reader asks:
`BM25Retriever` → `lexical_search`, `EnsembleRetriever` → `rank_fusion`, `CrossEncoderReranker` → the reranker switch.)

**D22. RAFT is decided by the evaluation with a fixed rule (§11.6), not by taste.** The research predicts it will be needed.

**D23. One JSON log line per answer** with ids, scores, route, per-stage milliseconds, token counts, stop reason, guard flag and
index version; the question appears only as its length and an 8-hex hash. Every key is spelled out (§7.8).

**D24. Concurrency and tail latency:** retrieval runs inside the existing single inference lock (2 vCPU, no contention between ONNX
and torch); one uvicorn worker (each worker would load its own copy). The reference client gives up at 60 s, queue wait included,
so (a) every answer stops at `answer_time_limit_seconds` (25 s; `stop_reason` "time_limit", logged) and (b) `max_requests_in_line`
is set so that (cap − 1) x live p95 + p95 ≤ 60 s: cap 2 at the 22 s gate, cap 3 at the 16 s target (§10.3 step 6; env
`GOT_MAX_REQUESTS_IN_LINE`, so the cap changes without a code publish). Rejected: keeping 4 in line (the 4th caller would wait about
66 s and time out while the server still spends CPU on it).

## 5. Architecture

### 5.1 Packages and dependency direction

```
retrieval_contracts   (frozen, standard library only)            ← imported by everything below
retrieval_encoders    (numpy, onnxruntime, tokenizers, hub)      ← index builder, service, evaluation
gpt2_demo_service
  language_models     (torch, tiktoken)                          no retrieval imports, no settings imports
  index_loading       (numpy, hub, contracts)                    download + validate + load the bundle; no torch, no pydantic
  index_search        (numpy, contracts, index_loading, encoder protocols)   no FastAPI, no torch, no pydantic
  grounded_answers    (index_search, language_models, tiktoken)  no FastAPI, no pydantic
  monitoring          (standard library, psutil)
  web_api             (FastAPI + pydantic schemas; the ONLY FastAPI user; app_factory wires everything)
  service_settings    (standard library + the three runtime settings dataclasses)   imported only by web_api.app_factory
retrieval_index_builder (offline, Mac)  → contracts, encoders     never imports gpt2_demo_service
retrieval_evaluation    (offline, Mac)  → contracts, encoders, gpt2_demo_service.{service_settings, index_loading, index_search,
                                          grounded_answers, language_models}; never imports retrieval_index_builder
```

Rules: no import cycles (service_settings imports the runtime dataclasses, never the reverse); every `__init__.py` is a bare
package marker that re-exports nothing (callers import from the module that defines a name). The answer locator both offline
packages need is in the frozen contract (§6.6), so WP1 and WP3 never import each other.

`retrieval_contracts` and `retrieval_encoders` live inside the service folder because they ship in the Space image; the offline
packages put `../gpt2_service_huggingface_space` on their import path (pytest `pythonpath`, `PYTHONPATH` for the commands).

### 5.2 Data flow

```
OFFLINE (Mac, retrieval_index_builder)
  raw books (heading lines) ──► chapter headings in reading order ─┐
  book_train_parts.json + books_validation_text.txt ──► 7 part texts ─► chapter alignment ─► ChapterRecord[]
  part texts + chapters ──► sentence-aligned chunks ──► PassageRecord[] (~12k)
  question_answer_train.json ──► rare-shingle place (in window, else corpus-wide) → part → chapter ──► QuestionAnswerRecord[] (34,001)
  records ──► compose_indexed_*_text ──► PinnedModelTextEncoder ──► vectors.npy
                                     └──► BM25 postings ──► lexical .npz
  everything ──► manifest.json (checksums, encoder fingerprint, settings) ──► self-verify
  ──► upload to PRIVATE dataset repo (refuse if not private) ──► commit hash = index revision

SPACE STARTUP (lifespan, in order)
  torch threads → weights (unchanged) → GoT + spam models
  → if retrieval enabled: snapshot_download(index repo, pinned revision) → load encoder (baked files, hash ONNX)
  → find_bundle_problems + fingerprint check → load records, vectors, postings → build EvidenceRetriever → one warm-up query
  → any failure: retrieval off with a reason; the service still starts

PER GAME OF THRONES REQUEST (inside the inference lock)
  question ─► encode [question, instruction + question] (one batch)
     pairs:    dense q-q top 10 + lexical(q+a) top 10 ─► RRF ─► keep similarity ≥ pair threshold ─► ≤ 2 examples
     passages: dense top 50 + lexical top 50 ─► RRF ─► top 20 ─► (rerank the top 10 → order) ─► gate on best score ─► ≤ 3 passages
  ─► answer_from_evidence: pack into the chosen layout within the budget ─► KV-cache generation (25 s limit) ─► verbatim guard
  ─► sources (included items only; one book snippet) ─► route of the packed prompt ─► log line
  ─► {"completion", "retrieval_used", "sources"}
```

Routes (`EvidenceRoute`): `passages_and_pairs`, `passages_only`, `pairs_only`, `no_evidence`, `retrieval_off`
(disabled, unavailable or `use_retrieval: false`), `retrieval_failed` (exception in this request). `RetrievedEvidence.route` says
what retrieval found; `GroundedAnswer.route` says what the packed prompt actually holds (a layout may ignore one item kind), and
`retrieval_used` is true exactly when at least one passage or pair is in the prompt.

## 6. The contract (frozen code in `retrieval_contracts/`)

Read the code; this section says what each piece guarantees.

### 6.1 Records (`bundle_records.py`)
- `PassageRecord(passage_id, part_number, book_title, series_number, chapter_number, chapter_title, character_start, character_end, token_count, text)`.
  Invariant: `part_text[character_start:character_end] == text`. `passage_id` = `"bp-<part>-<chapter 3 digits>-<chunk 3 digits>"`
  (chapter `xxx` when unknown), unique per bundle. `token_count` = GPT-2 tokens of `text`.
- `QuestionAnswerRecord(pair_id, source_index, question, answer, book_title, series_number, chapter_number, chapter_title)`;
  book and chapter from the pair's rare-shingle place (§6.6, §8.5); without one, chapter fields only when the pair's
  ±20,000-character location window lies inside one chapter (rare: the window is 40k wide, a chapter averages about 28k).
- `ChapterRecord(part_number, chapter_number, chapter_title, character_start, character_end)`; spans start AFTER the inline
  heading; a chapter cut by a part boundary (Feast parts 3 → 6 → 4) gets one record per part with the same number.
- `write_json_lines` / `read_json_lines`: sorted keys, UTF-8, strict keys (a format drift raises `ValueError`).
- Line i of a records file is row i of its vector file.

### 6.2 Book parts (`book_parts.py`)
Seven parts: 0 Dance, 1 Storm, 2 Game, 3 Feast-before, 4 Feast-after, 5 Clash, 6 the held-out Feast stretch. Publication order
(`series_number`): 1 Game, 2 Clash, 3 Storm, 4 Feast, 5 Dance. `locate_cleaned_offset(location_book_index, offset, part_lengths)`
maps a pair's `book_location` into `(part_number, offset_in_part)` or None (trimmed matter); tested at the Feast boundaries.

### 6.3 Bundle layout and manifest (`bundle_manifest.py`)

| File | Content |
|---|---|
| `manifest.json` | written last; format version 1, `index_version`, build time, builder source checksum, source data checksums, encoder fingerprint, chunking and lexical settings, counts, a checksum and size per data file |
| `book_passages.jsonl` | PassageRecord lines |
| `book_passage_vectors.npy` | float32 (passages, 384), unit rows, of `compose_indexed_passage_text` |
| `book_passage_lexical_index.npz` | lexical postings of `compose_indexed_passage_text` |
| `question_answer_pairs.jsonl` | QuestionAnswerRecord lines |
| `question_vectors.npy` | float32 (pairs, 384) of the question alone |
| `question_answer_lexical_index.npz` | lexical postings of `compose_indexed_pair_text` (question + answer) |
| `chapters.jsonl` | ChapterRecord lines |

Lexical `.npz` arrays (no pickles): `terms` (unicode, sorted ascending), `posting_starts` (int64, terms + 1, term i's postings are
`[starts[i], starts[i+1])`), `posting_rows` (int32, ascending within a term), `posting_weights` (float32, the full BM25 term weight).
Query score of document d = sum over DISTINCT query tokens t present in `terms` of weight(t, d); documents with score 0 are not
lexical hits. `index_version` = `"got-retrieval-<yyyymmdd of built_at_utc>-<first 8 hex of sha256(the 7 data-file sha256 hex
strings joined with no separator, in manifest order, as ASCII)>"`, so two bundles that differ in any file (vectors, pairs,
encoder) never share a version. Comment-only change in `bundle_manifest.py`, no format bump; the fixture manifest carries the value
of this rule (`got-retrieval-20261024-eeb6e60a`).
`find_bundle_problems(folder, manifest)` returns an empty list or the reasons (format version, file list, sizes, checksums).
The service additionally refuses: `lexical.tokenizer_version != LEXICAL_TOKENIZER_VERSION`, `chunking.indexed_text_version !=
INDEXED_TEXT_VERSION`, encoder fingerprint not equal to the loaded encoder's, row counts not equal to the manifest counts.

### 6.4 Text rules (`lexical_tokens.py`, `indexed_text.py`)
- `split_lexical_tokens`: NFKC, lower case, curly apostrophes straightened, words `[a-z0-9]+('[a-z0-9]+)*`, stopwords dropped
  before stemming, possessive `'s` off, `ies → y`, plain plural `s` off (not after `ss`, `us`, `is`). Version 1.
- `compose_indexed_passage_text(record)` = `"<book>, <chapter label>: <text>"`; `compose_indexed_pair_text` = `"<question> <answer>"`.
- `format_chapter_label(12, "Tyrion")` = `"Tyrion (chapter 12)"`; prologue / epilogue without a number; unknown = None.

### 6.5 Encoders (`encoder_contracts.py`)
- `TEXT_ENCODER_FILES`, `PASSAGE_PAIR_SCORER_FILES`: Hub id, full commit, `onnx/model.onnx`, `tokenizer.json`, 512 input tokens
  (both files were checked to exist at those commits on 2026-10-24).
- `PASSAGE_QUERY_INSTRUCTION` = `"Represent this sentence for searching relevant passages: "`, only before a visitor question
  for passage search.
- `EncoderFingerprint(model_id, revision, model_file_sha256, output_dimensions, pooling, passage_query_instruction)`.
- `TextEncoder` protocol: `fingerprint` property; `encode_texts(texts) -> float32 (n, d)` unit rows, no prefix added by the encoder.
- `PassagePairScorer` protocol: `score_pairs(question, passage_texts) -> float32 (n,)`, higher fits better (raw logits).

### 6.6 Answer locator (`rare_shingle_location.py`, added in part 6)
One rule for "where in the books is this text", shared by WP1 (pair chapters) and WP3 (gold spans, relevance), standard library only.
- A shingle = 4 consecutive ASCII words, lower case, apostrophes straightened, stopwords KEPT; matched on the original part text,
  so every offset points into it. Rare = at most 3 occurrences across all seven parts (almost every 4-word shingle is rare).
- `build_rare_shingle_index(part_texts) -> RareShingleIndex` (about 5 s and 1 GB for the real books).
- `locate_text(index, text, search_window: SearchWindow | None = None) -> LocatedSpan | None`: the densest cluster of the text's
  distinct rare shingles whose hits start within 1,000 characters of the cluster's first hit, at least 2 hits, ties by earliest
  part and start. `LocatedSpan(part_number, character_start, character_end, hits: tuple[ShingleHit, ...])`.
- Self-tests: two fixture pair answers land inside their gold sentences; a wrong-part window or a single hit gives None.

## 7. Runtime design (WP2, `gpt2_demo_service/`)

### 7.1 Settings (declared per package, every default in `service_settings.py`)

Each runtime package declares the frozen dataclass it consumes, with NO default values, next to its code:
```python
# index_search/evidence_search_settings.py
@dataclass(frozen=True) class EvidenceSearchSettings:
    passage_search_mode: Literal["hybrid", "dense_only", "lexical_only"]
    dense_candidate_count: int; lexical_candidate_count: int; fusion_constant: int; fused_candidate_count: int
    reranker_enabled: bool; rerank_candidate_count: int
    passage_evidence_floor: float           # dense cosine of the best candidate (calibrated)
    rerank_evidence_floor: float            # best rerank score, used only when the reranker is on (calibrated)
    max_passages: int; pair_candidate_count: int
    pair_example_threshold: float           # question-to-question cosine (calibrated leave-one-out)
    max_pair_examples: int
# grounded_answers/answer_settings.py
class PromptLayout(StrEnum): QUESTION_ONLY, PASSAGES_BEFORE_TEMPLATE, PAIR_EXAMPLES_BEFORE_QUESTION, PASSAGES_AND_PAIR_EXAMPLES
@dataclass(frozen=True) class GroundedAnswerSettings:
    layout: PromptLayout; context_token_budget: int; safety_margin_tokens: int
    answer_token_floor: int; answer_token_ceiling: int      # answer reserve = min(max(request tokens, floor), ceiling)
    book_snippet_enabled: bool
# web_api/request_limits.py
@dataclass(frozen=True) class PublicLimitSettings:
    max_requests_in_line: int; max_requests_per_visitor: int; rate_window_seconds: int; busy_retry_seconds: int
    rate_retry_seconds: int; got_question_max_characters: int; email_max_characters: int
```
`gpt2_demo_service/service_settings.py` is the ONE place for default values and the only reader of the environment:
```python
@dataclass(frozen=True) class IndexSourceSettings:          # where the bundle and the encoder files come from
    enabled: bool = True                                    # env GOT_RETRIEVAL_ENABLED (the kill switch)
    index_repo_id: str = "karamAM/gpt2-got-retrieval-index"
    index_revision: str = "<40-char commit, filled by the integrator>"   # env GOT_RETRIEVAL_INDEX_REVISION
    model_files_folder: Path = Path("/home/user/model_files")          # baked encoder files
    encoder_thread_count: int = 2
    scorer_input_token_limit: int = 256                     # reranker input cap in word pieces (D8)
@dataclass(frozen=True) class ModelSettings:                # the two GPT-2 models
    torch_thread_count: int = 2                             # env TORCH_THREADS (existing)
    weights_repo_id: str = "karamAM/gpt2-got-and-spam-weights"   # env WEIGHTS_REPO_ID (existing)
    got_weights_file_name: str = "got_model.pth"            # env GOT_WEIGHTS_FILE_NAME (a RAFT model switches here)
    spam_weights_file_name: str = "spam_model.pth"
    answer_time_limit_seconds: float = 25.0                 # D24
DEFAULT_EVIDENCE_SEARCH = EvidenceSearchSettings(
    passage_search_mode="hybrid", dense_candidate_count=50, lexical_candidate_count=50, fusion_constant=60,
    fused_candidate_count=20, reranker_enabled=False,       # env GOT_RETRIEVAL_RERANKER_ENABLED
    rerank_candidate_count=10, passage_evidence_floor=0.60, rerank_evidence_floor=0.0, max_passages=3,
    pair_candidate_count=10, pair_example_threshold=0.85, max_pair_examples=2)   # floors / threshold / switches from §11.6
DEFAULT_GROUNDED_ANSWERS = GroundedAnswerSettings(
    layout=PromptLayout.PASSAGES_AND_PAIR_EXAMPLES,         # env GOT_PROMPT_LAYOUT; default from §11.6
    context_token_budget=600, safety_margin_tokens=8, answer_token_floor=120, answer_token_ceiling=200,
    book_snippet_enabled=True)                              # env GOT_BOOK_SNIPPET_ENABLED
DEFAULT_PUBLIC_LIMITS = PublicLimitSettings(
    max_requests_in_line=3,                                 # env GOT_MAX_REQUESTS_IN_LINE; set from the live p95 (§10.3 step 6)
    max_requests_per_visitor=20, rate_window_seconds=60, busy_retry_seconds=15, rate_retry_seconds=30,
    got_question_max_characters=500, email_max_characters=2000)
@dataclass(frozen=True) class ServiceSettings:
    index_source: IndexSourceSettings; models: ModelSettings; evidence_search: EvidenceSearchSettings
    answers: GroundedAnswerSettings; limits: PublicLimitSettings
def load_service_settings(environment: Mapping[str, str]) -> ServiceSettings   # defaults + only the env names above; bad value -> ValueError naming the variable, at startup
def read_hub_token(environment: Mapping[str, str]) -> str | None                # HF_TOKEN; never stored in ServiceSettings, never logged
```
`load_service_settings({})` returns exactly the shipped defaults (the evaluation starts from it and varies fields with
`dataclasses.replace`). Inside the service only `web_api/app_factory.py` imports this module. If the file passes 150 lines, the
environment parsing moves to a sibling `settings_from_environment.py` (still the only environment reader).

### 7.2 Language models (`language_models/`)
- `decoding_cache.py`: `LayerKeyValueCache(batch, heads, capacity, head_width, dtype)` preallocates keys and values of shape
  `(batch, heads, capacity, head_width)` once; `append(new_keys, new_values) -> (all_keys, all_values)` writes the new positions into
  the next slice and returns VIEWS of the filled part (never `torch.cat`); writing past `capacity` raises. `DecodingCache(layer_count,
  batch, heads, capacity, head_width, dtype)` with `layers` and `cached_position_count`; the generator sizes it to prompt +
  max_new_tokens (≤ 1,024; about 200 MB at full length). One per request, discarded after it.
- `gpt_blocks.py`: `MultiHeadAttention.forward(x, layer_cache: LayerKeyValueCache | None = None)`; with a cache, the queries of the
  new positions attend to cached + new keys under `mask[past:past+n, :past+n]`. `TransformerBlock.forward(x, layer_cache=None)`.
  Attribute names and the persistent `mask` buffer unchanged.
- `gpt_model.py`: `GPTModel.forward(token_ids, decoding_cache: DecodingCache | None = None, last_position_only: bool = False) -> Tensor`;
  positions start at `cached_position_count`; blocks iterated explicitly (the `trf_blocks` container name stays). Default call is
  identical to today (the spam classifier relies on it).
- `token_sampling.py`: `pick_next_token(logits, temperature, top_k, top_p, random_generator: torch.Generator | None = None)`; repetition
  penalty still only over generated ids (copying names from the passages is not penalised).
- `got_text_generator.py`:
  ```python
  @dataclass(frozen=True) class SamplingSettings: temperature: float; top_p: float; top_k: int; repetition_penalty: float
  StopReason = Literal["end_token", "stop_marker", "token_limit", "time_limit"]
  @dataclass(frozen=True) class GeneratedAnswer: text: str; answer_token_count: int
      stop_reason: StopReason; prefill_milliseconds: float; decode_milliseconds: float
  class PromptTooLongError(ValueError)
  def load_got_text_generator(weights_path: Path, answer_time_limit_seconds: float) -> GotTextGenerator   # reads the 1.7 GB weights
  class GotTextGenerator:
      def __init__(self, model: GPTModel, tokenizer: tiktoken.Encoding, answer_time_limit_seconds: float) -> None   # tests inject a tiny random model
      # .tokenizer, .context_length (from the model, 1,024 in production)
      def generate_answer(self, prompt_token_ids: Sequence[int], max_new_tokens: int, sampling: SamplingSettings,
                          stop_strings: Sequence[str], random_generator: torch.Generator | None = None) -> GeneratedAnswer
  ```
  Prefill once, then one token per step with the cache; stop on 50256, on `"###"`, on `"Answer the following question"` (the model may
  start a new example after few-shot prompts), on a request stop text, or when prefill + decode time reaches the time limit (the text
  so far is returned, `stop_reason` "time_limit"); raises `PromptTooLongError` when prompt + max_new_tokens > 1,024. The time limit is
  a constructor argument (no settings import in this package); the evaluation passes a large value so seeded runs stay comparable.
- `spam_classifier.py`, `weights_download.py` move here unchanged except imports; repo id and file names arrive as arguments.

### 7.3 Prompt layouts (`grounded_answers/prompt_layouts.py`; `PromptLayout` itself is in `answer_settings.py`)
`TRAINING_PROMPT_HEADER` = `"Answer the following question with an appropriate response. Write a response that appropriately completes the qusetion."`
`render_question_block(question)` = header + `"\n\n### question:\n" + question + "\n\n### Response:\n"` (byte-identical to today).
`render_pair_example(question, answer)` = `render_question_block(question) + answer` (no end token; examples are joined by `"\n\n"`).
Inputs are given best first; rendering puts the best one LAST, next to the question.

| `PromptLayout` | Rendered text | Why it might work |
|---|---|---|
| `question_only` | question block | today's prompt; the fallback and the baseline |
| `passages_before_template` | passage₃ \n\n passage₂ \n\n passage₁ \n\n question block | the model read plain book blocks in training (raw passage text, no heading) |
| `pair_examples_before_question` | example₂ \n\n example₁ \n\n question block | closest to the trained format (74-token examples) |
| `passages_and_pair_examples` | passages \n\n examples \n\n question block | both signals; the in-format example sits nearest |

`render_prompt(layout, question, passage_texts, pair_examples) -> str`. A layout that does not use an item kind ignores it, and the
packer then does not list it as a source. A RAFT layout is added only if §11.6 says so (§12).

### 7.4 Packing (`grounded_answers/context_packing.py`)
```python
@dataclass(frozen=True) class PromptBudget: context_length: int; answer_reserve_tokens: int; context_token_budget: int
    safety_margin_tokens: int
@dataclass(frozen=True) class PackedPrompt: layout: PromptLayout; prompt_text: str; prompt_token_ids: tuple[int, ...]
    passages: tuple[ScoredPassage, ...]; pair_examples: tuple[ScoredPair, ...]      # included only, best first
    @property has_context -> bool
def trim_to_sentence_budget(text: str, max_tokens: int, tokenizer: tiktoken.Encoding) -> str
def pack_prompt(question: str, evidence: RetrievedEvidence, layout: PromptLayout, budget: PromptBudget,
                tokenizer: tiktoken.Encoding) -> PackedPrompt
```
Algorithm: base = question-only token count; allowance = min(context_token_budget, context_length − answer_reserve − margin − base);
take pair examples best first, then passages best first, each WHOLE while the running estimate fits; the first item that does not
fit is trimmed at a sentence end to the remaining allowance (kept only if at least 40 tokens remain) and packing stops. A passage is
skipped when more than half of the shorter of it and an already chosen passage of the same part overlap (adjacent chunks share one
sentence and stay allowed). Render; encode; while the real count breaks either limit, drop the lowest-ranked item and re-render.
Invariants (property-tested): never over 1,024 with the reserve; question text present unchanged; never more than `max_passages` /
`max_pair_examples`; only the last included item can be shorter than its record.

### 7.5 Index loading (`index_loading/`) and index search (`index_search/`)
```python
# index_loading/bundle_download.py
def download_index_bundle(repo_id: str, revision: str, token: str | None) -> Path      # snapshot_download(repo_type="dataset", revision, allow_patterns=the 8 names)
# index_loading/bundle_loading.py
class BundleValidationError(Exception)                                                # message = reasons, never text
@dataclass(frozen=True) class LexicalPostings: term_rows: Mapping[str, int]; posting_starts; posting_rows; posting_weights; document_count: int
@dataclass(frozen=True) class LoadedIndexBundle: manifest: BundleManifest; passages: tuple[PassageRecord, ...]; passage_vectors
    passage_lexical: LexicalPostings; pairs: tuple[QuestionAnswerRecord, ...]; question_vectors; pair_lexical: LexicalPostings
    chapters: tuple[ChapterRecord, ...]
def load_lexical_postings(npz_path: Path) -> LexicalPostings
def load_index_bundle(bundle_folder: Path, text_encoder_fingerprint: EncoderFingerprint) -> LoadedIndexBundle
# index_search/lexical_search.py / index_search/dense_search.py
@dataclass(frozen=True) class RankedRow: row: int; score: float
def score_lexical_matches(postings: LexicalPostings, text: str) -> NDArray[float32]          # (document_count,)
def find_top_lexical_matches(postings, text, limit) -> list[RankedRow]                      # score > 0, score desc, ties by row
def find_top_dense_matches(vectors, query_vector, limit) -> list[RankedRow]                 # argpartition, score desc, ties by row
# index_search/rank_fusion.py
@dataclass(frozen=True) class FusedCandidate: item_id: str; fused_score: float; ranks: tuple[int | None, ...]
def fuse_rankings(ranked_id_lists: Sequence[Sequence[str]], fusion_constant: int) -> list[FusedCandidate]  # 1-based ranks, ties by id
# index_search/evidence_search_settings.py: EvidenceSearchSettings (§7.1)
# index_search/evidence_records.py
class EvidenceRoute(StrEnum): PASSAGES_AND_PAIRS, PASSAGES_ONLY, PAIRS_ONLY, NO_EVIDENCE, RETRIEVAL_OFF, RETRIEVAL_FAILED
@dataclass(frozen=True) class ScoredPassage: record: PassageRecord; fused_rank: int; dense_score: float; lexical_score: float; rerank_score: float | None
@dataclass(frozen=True) class ScoredPair: record: QuestionAnswerRecord; question_similarity: float; fused_rank: int
@dataclass(frozen=True) class RetrievedEvidence: route: EvidenceRoute
    passage_candidates: tuple[ScoredPassage, ...]   # ranked, before the gate, ≤ fused_candidate_count (the evaluation reads these)
    passages: tuple[ScoredPassage, ...]             # after the gate, ≤ max_passages, best first
    pair_candidates: tuple[ScoredPair, ...]; pair_examples: tuple[ScoredPair, ...]
    best_evidence_score: float | None; stage_milliseconds: Mapping[str, float]
def build_empty_evidence(route: EvidenceRoute) -> RetrievedEvidence      # retrieval_off / retrieval_failed: nothing found, no stages
# index_search/evidence_retriever.py
class EvidenceRetriever:
    def __init__(self, bundle: LoadedIndexBundle, text_encoder: TextEncoder, settings: EvidenceSearchSettings,
                 pair_scorer: PassagePairScorer | None = None) -> None
    def retrieve(self, question: str) -> RetrievedEvidence
```
The gate score is mode-independent: dense cosines are computed for the candidates even in `lexical_only` mode. With the reranker on,
only the top `rerank_candidate_count` fused candidates are scored and reordered; the rest keep their fused order behind them.
Stage names in `stage_milliseconds`: `encode`, `dense_search`, `lexical_search`, `fusion`, `rerank`, `gate`.

### 7.6 Grounded answers (`grounded_answers/`)
```python
# answer_settings.py: PromptLayout, GroundedAnswerSettings (§7.1)
# source_citations.py
MAX_SNIPPET_WORDS = 25
@dataclass(frozen=True) class SourceCitation: source_type: SourceType; book: str | None; chapter: str | None; snippet: str | None
def choose_book_snippet(passage_text: str, max_words: int = MAX_SNIPPET_WORDS) -> str   # first ≤ 25 words from the passage start, "…" when cut; no question input
def build_source_citations(packed: PackedPrompt, book_snippet_enabled: bool) -> tuple[SourceCitation, ...]
    # passages best first, then pairs; ONLY the first passage gets a book snippet (when enabled), other passages snippet None;
    # a pair's snippet = its question cut to ≤ 25 words; chapter = format_chapter_label(...) or None
# verbatim_guard.py
MAX_SHARED_WORDS = 25
@dataclass(frozen=True) class GuardedCompletion: text: str; was_cut: bool; longest_shared_run: int
def limit_verbatim_overlap(completion: str, passage_texts: Sequence[str], max_shared_words: int = MAX_SHARED_WORDS) -> GuardedCompletion
# grounded_answer_service.py
@dataclass(frozen=True) class AnswerRequest: question: str; max_new_tokens: int; sampling: SamplingSettings
    stop_strings: tuple[str, ...]; use_retrieval: bool
@dataclass(frozen=True) class GroundedAnswer: completion: str; retrieval_used: bool; sources: tuple[SourceCitation, ...]
    route: EvidenceRoute; prompt_token_count: int; answer_token_count: int; stop_reason: StopReason
    stage_milliseconds: Mapping[str, float]; verbatim_guard_cut: bool
    passage_ids: tuple[str, ...]; pair_ids: tuple[str, ...]; best_evidence_score: float | None   # for the log line and the evaluation
class GroundedAnswerService:
    def __init__(self, generator: GotTextGenerator, retriever: EvidenceRetriever | None, settings: GroundedAnswerSettings) -> None
    def answer_question(self, answer_request: AnswerRequest, random_generator: torch.Generator | None = None) -> GroundedAnswer
        # = retrieve (or build_empty_evidence: retrieval_off / retrieval_failed), then answer_from_evidence with settings.layout and
        #   settings.context_token_budget
    def answer_from_evidence(self, answer_request: AnswerRequest, evidence: RetrievedEvidence, layout: PromptLayout,
                             context_token_budget: int, random_generator: torch.Generator | None = None) -> GroundedAnswer
        # pack -> generate -> verbatim guard (packed passages) -> sources -> route; the ONE answer path the service and the evaluation share
```
Route of an answer: `evidence.route` when it is `retrieval_off` or `retrieval_failed`; otherwise from what the packed prompt holds
(`passages_and_pairs`, `passages_only`, `pairs_only`, `no_evidence`). Verbatim rule: compare lower-cased word sequences; at the first
point where a shared run would pass 25 words, cut the completion back to the last sentence end before that point, or, without one,
after the 25th shared word with "…". The runtime guard reads the packed passages only; A16 (offline) checks every passage (D18).

### 7.7 Web API (`web_api/`)
- `api_schemas.py` (pydantic): `GenerateRequest` = today's fields + `use_retrieval: bool = True`;
  `SourcePayload{source_type: str, book: str | None, chapter: str | None, snippet: str | None}` (all four keys always present);
  `CompletionResponse{completion: str}` (spam); `GroundedCompletionResponse{completion: str, retrieval_used: bool, sources:
  list[SourcePayload]}` (Game of Thrones). No `response_model_exclude_none` anywhere (D17).
- `generate_route.py`: the `/generate` route is declared with `response_model=None`, returns one of the two models explicitly, and
  documents both shapes in `responses=` (OpenAPI still shows them).
- `app_factory.py`: `ServiceComponents` (answer service, spam classifier, retrieval status, weights info, index version),
  `load_service_components(settings: ServiceSettings, hub_token: str | None) -> ServiceComponents`,
  `create_app(settings: ServiceSettings, component_loader=load_service_components) -> FastAPI` (tests inject fakes, no 1.7 GB
  weights), `create_production_app() -> FastAPI` for `uvicorn --factory` (the only caller of `load_service_settings(os.environ)` and
  `read_hub_token`). CORS list unchanged.
- `health_route.py`, `visitor_address.py`, `request_limits.py` (unchanged logic; `PublicLimitSettings` declared here, values from
  `service_settings`), `try_it_page.html` (shows the answer, then "From the books:" with each source's book and chapter, and the one
  snippet when present; same 500-character box).
- `/health` keeps every key and adds `"retrieval": {"ready", "index_version", "index_revision" (8 hex), "passages", "pairs",
  "text_encoder", "reranker_enabled", "prompt_layout", "problem"}`. The client example's `health.models.includes("got")` still works.
- Errors unchanged: 422 / 429 / 503; retrieval problems never become 500.
- `Server-Timing: retrieval;dur=…, prefill;dur=…, decode;dur=…` on Game of Thrones responses (no body change; the latency probe reads it).

### 7.8 Monitoring (`monitoring/`)
`memory_logging.py` (moved), `stage_timing.py` (`StageTimer.measure(stage_name)` context manager → milliseconds per stage),
`request_logging.py` (`log_answer_event(answer: GroundedAnswer, question: str, index_version: str | None) -> None`: one JSON line
`{"event": "got_answer", "question_characters", "question_hash", "route", "retrieval_used", "evidence_score", "passage_ids",
"pair_ids", "stage_milliseconds", "prompt_token_count", "answer_token_count", "stop_reason", "verbatim_guard_cut",
"index_version"}`; every key spelled out). It is also the fallback source of stage times if the Hugging Face edge strips
`Server-Timing` (§10.3 step 6).

## 8. Offline index builder (WP1, `retrieval_index_builder/` + `retrieval_encoders/`)

### 8.1 Encoders (`gpt2_service_huggingface_space/retrieval_encoders/`)
```python
def model_folder_for(files: PinnedModelFiles, base_folder: Path) -> Path              # base/<model id with "/" → "__">
def download_model_files(files: PinnedModelFiles, base_folder: Path) -> Path          # hf_hub_download of the ONNX + tokenizer files at the pinned revision
# command: python -m retrieval_encoders.model_file_download --target /home/user/model_files   (both models; the Dockerfile runs it)
def build_inference_session(model_path: Path, thread_count: int) -> onnxruntime.InferenceSession   # intra-op = thread_count, inter-op = 1, sequential
class PinnedModelTextEncoder:     # implements TextEncoder
    def __init__(self, base_folder: Path, files: PinnedModelFiles = TEXT_ENCODER_FILES, thread_count: int = 2, batch_size: int = 32)
    fingerprint -> EncoderFingerprint   # ONNX checksum computed once at load
    def encode_texts(self, texts: Sequence[str]) -> NDArray[float32]   # truncate 512, pad per batch, feed the inputs the graph declares, CLS row, unit length
class PinnedModelPairScorer:      # implements PassagePairScorer
    def __init__(self, base_folder: Path, input_token_limit: int, files: PinnedModelFiles = PASSAGE_PAIR_SCORER_FILES, thread_count: int = 2)
    def score_pairs(self, question: str, passage_texts: Sequence[str]) -> NDArray[float32]   # pair encoding (question, passage) cut to input_token_limit (≤ 512) word pieces, logit per row
```
Parity check (once, scratch venv, not a dependency): cosine ≥ 0.9999 between our vectors and the sentence-transformers reference
for 5 sample texts; same for the scorer's order on 5 pairs. Record the result in the builder's report.

### 8.2 Builder modules
```python
# index_build_settings.py
@dataclass(frozen=True) class IndexBuildSettings:
    source_data_folder: Path      # kaggle_finetune/mixed_run/staging/mixed_books_qa_data
    raw_books_folder: Path        # ~/Downloads/drive-download-20250818T222213Z-1-001
    output_folder: Path           # retrieval_index_builder/build_output/bundle (private, never committed)
    model_files_folder: Path      # retrieval_index_builder/build_output/model_files
    target_tokens: int = 200; max_tokens: int = 230; overlap_sentences: int = 1; minimum_tail_tokens: int = 40
    pair_window_characters: int = 20000                     # half-width of a pair's coarse location window
    index_repo_id: str = "karamAM/gpt2-got-retrieval-index"; embedding_batch_size: int = 64
# source_alignment/   (where text, chapters and pairs sit in the books)
def load_book_part_texts(source_data_folder: Path) -> dict[int, str]                    # 7 parts (part 6 from books_validation_text.txt)
RAW_BOOK_FILE_NAMES: Mapping[int, str]                                                    # location_book_index → raw file name
def read_raw_chapter_headings(raw_book_path: Path) -> list[str]                           # body headings in reading order (contents list skipped)
@dataclass(frozen=True) class ChapterAlignment: chapters: tuple[ChapterRecord, ...]; report: Mapping[str, object]
def align_chapters(part_texts: Mapping[int, str], headings_by_book: Mapping[int, list[str]]) -> ChapterAlignment
def find_chapter_at(chapters: Sequence[ChapterRecord], part_number: int, offset: int) -> ChapterRecord | None
@dataclass(frozen=True) class PairPlacement: part_number: int | None; chapter: ChapterRecord | None
    method: Literal["shingles_in_window", "shingles_corpus_wide", "window_inside_chapter", "window_only", "unplaced"]
def place_pair(question: str, answer: str, book_location: dict | None, part_lengths: Mapping[int, int],
               chapters: Sequence[ChapterRecord], shingle_index: RareShingleIndex, window_characters: int) -> PairPlacement   # §8.5
def build_pair_records(pairs_path: Path, part_texts: Mapping[int, str], chapters: Sequence[ChapterRecord],
                       settings: IndexBuildSettings) -> tuple[tuple[QuestionAnswerRecord, ...], Mapping[str, int]]   # records + placement counts by method
# passage_chunking/
def split_sentence_spans(text: str) -> list[tuple[int, int]]
def count_gpt2_tokens(text: str) -> int
def chunk_chapter(part_text: str, chapter: ChapterRecord, settings: IndexBuildSettings) -> list[PassageRecord]
def chunk_all_parts(part_texts, chapters, settings) -> tuple[PassageRecord, ...]
# bundle_writing/
def build_lexical_postings(documents: Sequence[str], saturation_k1: float = 1.2, length_normalization_b: float = 0.75) -> dict[str, NDArray]
def encode_in_batches(encoder: TextEncoder, texts: Sequence[str], batch_size: int) -> NDArray[float32]
def compose_index_version(built_at_utc: str, files: Sequence[BundleFileEntry]) -> str   # §6.3 rule; fixture → "got-retrieval-20261024-eeb6e60a"
def write_index_bundle(output_folder: Path, passages, pairs, chapters, encoder: TextEncoder, settings: IndexBuildSettings,
                       source_files: Sequence[Path]) -> BundleManifest       # writes 7 files, then the manifest, then find_bundle_problems must be []
@dataclass(frozen=True) class UploadReceipt: repo_id: str; commit_hash: str; index_version: str
def upload_index_bundle(bundle_folder: Path, repo_id: str, token: str) -> UploadReceipt  # create_repo(dataset, private, exist_ok); REFUSE unless repo_info(...).private
# build_index_command.py
#   python -m retrieval_index_builder.build_index_command [--upload]
#   python -m retrieval_index_builder.build_index_command --target-tokens 128 --max-tokens 150 --output-folder build_output/ablation_bundles/target_128
#   (ablation builds: never uploaded; 128/150 and 256/290 besides the default 200/230)
```

### 8.3 Chapter alignment (the hard part)
- Headings: read each raw file's lines; heading lines are short upper-case point-of-view names, PROLOGUE / EPILOGUE, or
  "CHAPTER <WORD>" followed by a name line (Clash). Skip the contents list (the first run of headings before the body text; for the
  files that have one it repeats the same sequence). Game's raw file has broken first letters ("ATELYN", "ROLOGUE"): match by the
  contents list order there.
- Alignment in the cleaned part texts: scan forward only, for each expected heading a regex that allows one optional space
  between any two letters (`P\s?R\s?O\s?L\s?O\s?G\s?U\s?E`, "C ATELYN", "PROLOGU E"), upper case, followed by a space and a word
  that starts upper case or a split capital ("I t was", "D ragons"); accept the first hit at least 3,000 characters after the
  previous chapter start. The Feast parts are scanned as one sequence 3 → 6 → 4.
- Expected chapter counts (check against the raw contents lists, which win): Game 73, Clash 70, Storm 82, Feast 46, Dance 73.
- Report per book: expected, found, unmatched headings, shortest and longest chapter. Build gate: ≥ 95 % of headings aligned per
  book; an unmatched stretch becomes part of the previous chapter. A book under 95 % ships with chapter fields None for its
  passages (citations show the book only), and the report says so.

### 8.4 Chunking rule
Sentence spans inside one chapter span; add sentences while the chunk stays ≤ `target_tokens`; a single sentence over
`max_tokens` is cut at word boundaries; the next chunk starts with the last `overlap_sentences` sentence(s) of the previous one
when those are ≤ 60 tokens; a tail under `minimum_tail_tokens` merges into the previous chunk if the result stays ≤ `max_tokens`.
`text` is always an exact substring of the part text. Expected about 12-14k passages; the report gives counts and token percentiles.

### 8.5 Pair placement (book and chapter of each pair)
One rule, built on the frozen locator (§6.6), over question + answer text:
1. Pair with a coarse `book_location` that maps into a part: `locate_text` inside the window offset ± 20,000 (clamped to the part).
   Found → part + the chapter holding the located span's centre (`shingles_in_window`). Never searched outside its window.
2. Pair without a usable location: `locate_text` over all parts (`shingles_corpus_wide`).
3. No shingle cluster but a window: chapter only when the whole window lies inside one chapter (`window_inside_chapter`), else book
   only (`window_only`). Nothing at all: `unplaced` (book and chapter None).
Measured with the locator (§2 fact 14): 17,488 in window, 2,221 corpus-wide, 875 window only, 13,417 unplaced, so about 58 % of
pairs can carry a chapter (versus almost none with the window rule alone). The build report gives the counts by method and the final
chapter rate per book.

## 9. File tree (end state)

```
GPT-2-GOT-DEPLOYMENT/
  rag-design-1024/                     architect: DESIGN.md, AGENTS.md
  gpt2_service_huggingface_space/      Space build context (WP2 owns all of it except the two packages marked)
    AGENTS.md README.md Dockerfile .dockerignore requirements.txt requirements-test.txt pytest.ini gpt2_space_client_example.mjs
    retrieval_contracts/               ARCHITECT, FROZEN (exists): __init__ bundle_records bundle_manifest book_parts
                                       lexical_tokens indexed_text encoder_contracts rare_shingle_location AGENTS.md
      shared_fixture/                  (exists) fixture_files fixture_encoders test_shared_fixture + 3 JSON + AGENTS.md
        fixture_bundle/                (exists) manifest + 7 data files + AGENTS.md
    retrieval_encoders/                WP1: __init__ model_file_download onnx_session_building pinned_text_encoder
                                       pinned_pair_scorer AGENTS.md
    gpt2_demo_service/                 WP2: __init__ service_settings AGENTS.md
      language_models/                 __init__ gpt_model gpt_blocks decoding_cache token_sampling got_text_generator
                                       spam_classifier weights_download AGENTS.md
      index_loading/                   __init__ bundle_download bundle_loading AGENTS.md
      index_search/                    __init__ evidence_search_settings lexical_search dense_search rank_fusion
                                       evidence_records evidence_retriever AGENTS.md
      grounded_answers/                __init__ answer_settings prompt_layouts context_packing source_citations verbatim_guard
                                       grounded_answer_service AGENTS.md
      monitoring/                      __init__ memory_logging stage_timing request_logging AGENTS.md
      web_api/                         __init__ api_schemas app_factory generate_route health_route visitor_address
                                       request_limits try_it_page.html AGENTS.md
    tests/                             WP2: AGENTS.md + language_models/ index_loading/ index_search/ grounded_answers/ web_api/
                                       (each with AGENTS.md)
  retrieval_index_builder/             WP1 (offline): __init__ index_build_settings build_index_command pytest.ini
                                       requirements.txt AGENTS.md
    source_alignment/                  __init__ book_part_texts raw_chapter_headings chapter_alignment pair_placement
                                       pair_records AGENTS.md
    passage_chunking/                  __init__ sentence_spans gpt2_token_counting chapter_chunking AGENTS.md
    bundle_writing/                    __init__ lexical_postings_writing vector_writing bundle_assembly bundle_upload AGENTS.md
    tests/                             AGENTS.md + test files
    build_output/                      generated, private, never uploaded except the bundle to the private repo; AGENTS.md
  retrieval_evaluation/                WP3 (offline): __init__ evaluation_settings run_evaluation_command pytest.ini
                                       requirements.txt AGENTS.md
    golden_sets/                       __init__ golden_question_records held_out_passage_questions fact_question_set
                                       fact_dev_question_set no_answer_questions regression_questions AGENTS.md
    metrics/                           pure scoring: __init__ ranking_metrics passage_relevance answer_overlap_scoring
                                       copyright_checks fact_rule_loading AGENTS.md
    experiments/                       runners: __init__ retrieval_ablation threshold_calibration answer_runs
                                       evaluation_report_writing AGENTS.md
    live_checks/                       __init__ live_contract_check live_latency_probe AGENTS.md
    tests/                             AGENTS.md + test files
    results/                           generated (ids, scores, model answers; no passage text); AGENTS.md
```
`.dockerignore` (WP2): `**/shared_fixture`, `**/tests`, `**/__pycache__`, `**/AGENTS.md`, so a local build matches the Space image.
The 13 flat service files move: `app.py` → `web_api/` (split into factory + routes), `request_limits.py` → `web_api/`,
`got_text_generator.py gpt_model.py gpt_blocks.py token_sampling.py spam_classifier.py weights_download.py` → `language_models/`,
`memory_logging.py` → `monitoring/`, `try_it_page.html` → `web_api/`. Every file ≤ 150 lines; every code line commented;
full-word names (domain words `got`, `gpt2`, `onnx` are fine; `rag`, `bm25`, `rrf`, `kv`, `mrr`, `ndcg`, `ms` are not used in
identifiers or keys: `mean_reciprocal_rank`, `normalized_discounted_cumulative_gain`, `stage_milliseconds`).

## 10. Deploy, rollout, rollback

### 10.1 Image (WP2 writes; the integrator verifies)
```
requirements.txt adds: numpy==2.4.4  onnxruntime==1.30.0  tokenizers==0.23.3
  (tokenizers 0.23.2 requires huggingface-hub < 2.0 and cannot install beside the pinned huggingface_hub==2.3.0; 0.23.3, released
   2026-10-09, allows hub < 3.0. Resolver check passed on 2026-10-24 with numpy 2.4.4, onnxruntime 1.30.0, torch 2.8.0+cpu,
   protobuf 7.36.2, re-run by the architect.)
Resolver gate (WP2 done-when, also the integrator before publishing):
  uv pip compile requirements.txt --python-version 3.11 --python-platform x86_64-manylinux_2_28 \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match      → must exit 0
Dockerfile:
  COPY requirements.txt → pip install (CPU torch index as today) → tiktoken warm-up (as today)
  COPY --chown=user retrieval_contracts ./retrieval_contracts
  COPY --chown=user retrieval_encoders ./retrieval_encoders
  RUN python -m retrieval_encoders.model_file_download --target /home/user/model_files     (public files, pinned commits)
  COPY --chown=user gpt2_demo_service ./gpt2_demo_service
  CMD ["uvicorn", "gpt2_demo_service.web_api.app_factory:create_production_app", "--factory",
       "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
.dockerignore: **/shared_fixture  **/tests  **/__pycache__  **/AGENTS.md
requirements-test.txt (not uploaded): pytest==9.1.1  httpx==0.28.1
```

### 10.2 Publish script (`~/Desktop/HQ/core/tools/huggingface/publish_gpt2_space.py`; integrator edits, it is an HQ file)
- `allow_patterns=["Dockerfile", "README.md", "requirements.txt", "gpt2_demo_service/*", "retrieval_contracts/*", "retrieval_encoders/*"]`
  (fnmatch `*` crosses folders), `ignore_patterns=["*/shared_fixture/*", "*/__pycache__/*", "*AGENTS.md", "tests/*", "*.pyc"]`,
  `delete_patterns=["*.py", "*.html"]` (removes the old top-level modules; files uploaded in the same commit are kept).
- After upload: list the Space files and fail if any old top-level `.py` remains or a fixture / test file is present.
- New `--rollback-folder <path>`: re-uploads a downloaded snapshot with the same delete patterns.

### 10.3 Rollout (integrator, after WP1-3 are green)
1. Build the real bundle, read the alignment and chunk reports, upload (private check), note the commit hash.
2. Evaluation Phase A on the Mac with the real bundle, the two ablation bundles and the local weights
   `kaggle_finetune/mixed_run_results/model.pth` (SHA-256 starts 528ef774 = live): retrieval ablations, calibration, answer layouts
   (about 2,450 generations, 2-6 hours, capped as in §11.5) → `recommended_settings.json` + report.
3. Apply §11.6: write the defaults into `service_settings.py` (bundle revision, layout, thresholds, switches).
4. Local run: `uvicorn --factory` with the real weights and bundle; contract check; latency on the Mac; resolver gate (§10.1).
5. Snapshot the live Space at commit `3103589f` (full hash from the Hub) into a private local folder; publish.
6. Wait for RUNNING; live checks: `/health` retrieval ready with the right index version; contract check; 20-call latency probe;
   memory; the 10 known questions live (one sample each) as a smoke test. First check that the Hugging Face edge passes
   `Server-Timing` through; if it does not, read stage times from the D23 log lines in the Space run log. Then set the line cap:
   the largest `max_requests_in_line` with (cap − 1) x live p95 + p95 ≤ 60 s (cap 2 at 22 s, cap 3 at 16 s, 4 only below 15 s),
   as the Space variable `GOT_MAX_REQUESTS_IN_LINE` and as the code default at the next publish.
7. Any gate fails → quick kill `GOT_RETRIEVAL_ENABLED=false` as a Space variable (restart about 35 s; answers become today's prompt,
   faster), or full rollback with `--rollback-folder` (code) / weights file setting (model).
8. Report results; Karam sees the live Space, the numbers and the rollback line.

## 11. Evaluation (WP3, `retrieval_evaluation/`)

### 11.1 Golden sets (gold stored as rare-shingle hits in the part text, never passage ids)
Tuning sets (choose settings) and reported sets (gates) never mix: G1 dev and FD tune; G1 test, G2 and G4 are reported, and G2/G4
run ONCE, on the no-retrieval baseline and on the configuration already chosen.
- **G1 held-out passage questions:** the 816 `answerable` pairs of `question_answer_validation.json`. Gold = `locate_text(index,
  question + " " + answer, SearchWindow(6, 0, len(part 6)))` from the frozen locator (§6.6): the cluster's hits are the gold
  (unwidened). Unlocated questions are dropped. **Block split** (the questions were generated one per 200-character chunk, so
  consecutive held-out questions sit a median 156 characters apart and alternate rows would put near-twins on both sides): cut part
  6 into 10 equal contiguous blocks (about 26,125 characters each); a question whose gold centre ((start + end) / 2) falls in an
  even block (0, 2, …) is dev, odd is test; questions within 2,000 characters of a block edge are dropped. Measured (§2 fact 14):
  443 located, 58 near an edge, **189 dev, 196 test**; the report repeats the counts. At about 196 questions, A5's hit@5 has a 95 %
  margin of about ±6 points; the report states it.
- **FD fact-dev (new, `golden_sets/fact_dev_question_set.py`):** about 40 new questions in the `FACT_QUESTIONS` format (question,
  accepted groups, rejected terms), scored by the same `is_right`. Written by WP3 across all five books and the kinds of facts
  visitors ask (who / what / where; sigils, seats, swords, wolves, titles, kin, events). Rules, each checked by a test: none is one
  of the 30 G2 questions or shares more than 0.6 token F1 with one; none is word for word (normalised exact match) in the 34,001
  pairs; every question carries `evidence_part_number` + `evidence_character_offset` (no book text) where WP3 checked, with a
  script, that one accepted term of each group appears within 2,000 characters in the cleaned books.
- **G2 facts (reported only):** the 30 `FACT_QUESTIONS` with `is_right`, loaded from
  `kaggle_finetune/mixed_run/mixed_kernel_source/fact_questions.py` (and `kernel_source/run_settings.py`) so scoring is the same
  function that produced 10/30; the first 10 are the known ten; the 3 word-for-word ones flagged by normalised exact match against
  the 34,001 pair questions (A2).
- **G3 no-answer:** about 40 written questions: 25 off-topic ("How many moons does Jupiter have?") and 15 series-sounding facts the
  books do not contain (later television-only events). They must get `retrieval_used: false` or evidence below the floor.
- **G4 regression (reported only):** "Who sits on the Iron Throne?" (rejected: gregor, clegane, mountain), "Who sits on the Iron
  Throne at the end of A Storm of Swords?" (accepted: tommen; rejected as above + stannis), "Who is called the Kingslayer?" (jaime;
  rejected stannis), "Where is Theon Greyjoy's family from?" (as in G2).
- G1 answer scores are inflated in both arms because the 2025 starting model read part 6 (§2 fact 13); the report says so.

### 11.2 Metrics (`metrics/`: pure functions, no model, no files)
- **Relevance (`passage_relevance.py`):** a passage is relevant to a G1 question when it contains at least one of the question's
  gold hits whole (same part, `passage.character_start ≤ hit.character_start` and `hit.character_end ≤ passage.character_end`). No
  widening, so bigger chunks get no free overlap.
- **Ranking (`ranking_metrics.py`):** `hit_at_k`, `precision_at_k`, `recall_at_k` (relevant retrieved / relevant passages in the
  index for that question), `mean_reciprocal_rank` at 10 (mean of 1 / rank of the first relevant passage),
  `normalized_discounted_cumulative_gain` at k (binary gain, discount 1/log2(rank + 1), ideal over min(relevant, k)), context
  precision (share of PACKED passages that are relevant), and **hit within the packed budget** (a relevant passage among the
  passages taken in rank order until `context_token_budget` tokens; the only measure used to compare chunk sizes).
- **Answers (`answer_overlap_scoring.py`, `fact_rule_loading.py`):** fact score (FD, G2), token F1 and key-term recall against the
  gold answer (G1), unsupported-name rate (capitalised names in the answer found in neither the packed context nor the question:
  the "Gregor on the throne" detector), no-evidence rate.
- **Copyright (`copyright_checks.py`, A16):** longest shared word run between every response field and ALL passages of the bundle
  (a 26-word shingle set over every passage; an implementation independent of the service's guard, on purpose), plus at most one
  book snippet per response and ≤ 25 words per snippet.

### 11.3 Retrieval ablations (`experiments/retrieval_ablation.py`; G1 dev, then the chosen one once on test)
dense only · lexical only · hybrid · hybrid + rerank (top 10); RRF constant 20 / 60 / 100; per variant hit@1/3/5/10, mean
reciprocal rank at 10, normalized discounted cumulative gain at 10 and stage milliseconds on the Mac (the Space numbers come from
the live probe). hit@k is compared only between variants of ONE chunk size. Chunk size (three bundles: 128/150, 200/230, 256/290,
built by WP1's command) is compared by hit within the packed budget at 300 / 450 / 600 tokens. Also: which questions the reranker
fixed or broke. Variants are `dataclasses.replace` of the shipped `EvidenceSearchSettings` (`load_service_settings({})`).

### 11.4 Calibration (`experiments/threshold_calibration.py`)
- **Evidence floor:** the highest floor at which ≥ 90 % of G1 dev questions AND ≥ 70 % of FD questions still get evidence (A7b);
  report the G3 acceptance at that floor (A7). The G1 questions reuse the passage's own words; FD keeps the floor honest for the
  general questions visitors ask.
- **Pair threshold, leave-one-out:** a seeded sample of 3,000 training pairs; each one's question is searched with its own
  `pair_id` removed from `pair_candidates` (retrieve with the shipped settings, `pair_example_threshold` set to −1 so every
  candidate is visible); the top remaining candidate's `question_similarity` is its score, and it counts as correct when token
  F1(candidate answer, own answer) ≥ 0.5. Choose the threshold on one seeded half (the lowest with precision ≥ 0.95; gate 0.90) and
  report precision and coverage on the other half (A8) and on FD (an example is correct there when its answer `is_right` under the
  FD rule). G2 is not used. (Held-out questions cannot calibrate this: the 325 pairs that share their passage are not in the index.)

### 11.5 Answer runs (`experiments/answer_runs.py`)
Every run goes through `GroundedAnswerService.answer_from_evidence` (§7.6): retrieval runs ONCE per question and the evidence is
reused across layouts and budgets; the runner never rebuilds routes, the guard or sources itself. Seeds: sample s of question q uses
`torch.Generator().manual_seed(1000 * s + q_index)`, identical across configurations; `max_new_tokens` as in the mixed run's fact
evaluation; the generator gets a time limit of 600 s so no seeded answer is cut by the clock.
- Selection grid: the four layouts of §7.3 at budget 600, then budgets 300 / 450 for the best layout (`question_only` once), on FD
  (3 samples) + all G1 dev (1 sample): about 6 x 309 = 1,850 generations.
- Final: baseline (`question_only`, no retrieval) and the chosen configuration on G2 + G4 (3 samples) + all G1 test (1 sample):
  about 2 x 298 = 600 generations.
- Estimate: about 2,450 generations at 3-8 s each on the Mac = about 2-5.5 hours. Cap: the runner times the first 30 generations
  and projects the total; above 6 hours it halves the G1 dev subset (seeded) and says so in the report; never more than 3,000
  generations in one run. Every answer record keeps the route, passage and pair ids, scores, token counts and stop reason (no
  passage text).

### 11.6 Decision rules (written into `recommended_settings.json` with the evidence)
1. `passage_search_mode`: hybrid unless a single search BEATS it by ≥ 2 points of G1 dev hit@5 (D7).
2. `reranker_enabled`: true only if +3 points G1 dev hit@3 and A9 holds on the Space estimate (Mac time x 3 as the planning factor,
   checked against D8's cost prior, confirmed live).
3. `layout` and `context_token_budget`: best FD majority score; ties by G1 dev key-term recall, then by fewer prompt tokens.
4. Ship gate, measured once on the chosen configuration: A1-A5, A7, A7b, A8, A13-A16 must hold; A9-A12 are confirmed live.
5. **RAFT needed** if the chosen configuration scores < 20/30 on G2 OR < 7/10 on the known ten OR its G1 test key-term recall gain
   over `question_only` is < 10 points. Phase A ships if it passes the ship gate; if it fails, ship the code with retrieval OFF
   (faster answers, same prompt) and run RAFT. (G2 is used here only as the reported outcome of one configuration, never to choose.)

### 11.7 Live checks (`live_checks/`)
`check_live_contract(space_url) -> list[str]`: the portfolio body `{prompt, max_new_tokens: 50, task_name: "got"}` → 200, string
`completion`, no `text`, keys exactly {completion, retrieval_used, sources}, each source exactly {source_type, book, chapter,
snippet}, at most one book snippet; spam body → exactly `{"completion"}`; 422 / 503 shapes unchanged. `probe_live_latency(space_url,
questions, call_count=20) -> LatencySummary` (p50 / p95 total and per Server-Timing stage; it reports when the header is missing).
Run only after a deploy, one caller, at most 20 calls (the Space is public and slow).

## 12. Phase B: RAFT grounding fine-tune (only if §11.6 rule 5 fires; separate package after WP1-3)
Folder `kaggle_finetune/retrieval_grounding_run/`, built like `mixed_run/` and reusing its modules.
- Examples (built on the Mac with the real retriever and `pack_prompt`, exported as token ids, so training and serving cannot drift):
  for each training pair with an oracle passage (a part 0-5 passage holding ≥ 2 of the pair's rare hits, §6.6): 80 % oracle +
  2 hard distractors (top retrieved passages without those hits), 20 % three distractors; pair examples from the retriever
  excluding the pair itself and near-duplicates (cosine ≥ 0.95); passage order shuffled with a seed; the chosen Phase A layout
  (if that was `question_only`, use `passages_and_pair_examples`). Never part-6 passages.
- Sequences are full packed prompts at the shipped budget, so they reach about 1,000 tokens and train the template at the positions
  retrieval puts it (§2 fact 3); batch 1-2 with gradient accumulation on the T4.
- Mix: 70 % retrieval examples, 15 % plain template pairs (keeps the no-evidence path), 15 % book blocks (keeps the voice); prompt
  masked (answer + end token only); start from the live mixed weights; one epoch on a free T4; in-notebook evaluation on exported
  packed prompts for FD and the G1 dev questions (G2 stays a reported set).
- Output `got_model_retrieval_grounded.pth` as a NEW file in the weights repo; switch with `GOT_WEIGHTS_FILE_NAME`; rollback =
  switch back. Same §11 evaluation before it goes live.

## 13. Errors and edge cases
| Case | Behaviour |
|---|---|
| Bundle download fails / checksum mismatch / version or fingerprint mismatch | retrieval off, `/health.retrieval.problem` says which, one ERROR log line, `/generate` works with today's prompt |
| Encoder files missing in the image | same as above (build bug; the integrator's post-publish check catches it) |
| Exception inside one retrieval | route `retrieval_failed`, plain prompt, log line with the exception type only |
| Prompt would exceed the budget | packer drops items; `PromptTooLongError` only on a programming error (tested) |
| Question with no lexical tokens (e.g. "???") | lexical list empty, dense still runs, gate decides |
| Model copies a packed passage | verbatim guard cuts at 25 shared words; logged |
| Answer reaches `answer_time_limit_seconds` | generation stops, the text so far goes through the guard and gets its sources as usual, `stop_reason` "time_limit" logged |
| Source with no known chapter | `chapter: null`, the key is still present |
| Line full (`max_requests_in_line`) | 503 with `Retry-After` as today |
| `use_retrieval: false` | route `retrieval_off`, today's prompt, `retrieval_used: false`, `sources: []` |
| Spam request | untouched path, body exactly `{"completion"}` |

## 14. Work packages (parallel, disjoint folders)

Common rules for every builder: read the project-root `CLAUDE.md` if any, the AGENTS.md of every folder you touch, and this file;
files ≤ 150 lines, a short comment on every code line, full-word names, typed public functions, frozen dataclasses for records,
an AGENTS.md in every folder you create, `__init__.py` files that re-export nothing, no dead code, no book text in any test, never
print or write the Hugging Face token, never push to any GitHub main, no browser. Venv: `uv venv ~/.venvs/got-retrieval-<package>
--python 3.11` (one per package so parallel builders never fight over one). After each main item ONE run-record line:
`python3 ~/Desktop/HQ/_shared/bin/runlog.py hq step wf-1024-2000-got-rag '<what you finished>' --part <your part number>`.
The contract and the fixture are read-only for all three; a disagreement goes to the integrator, not into the fixture. No package
imports another package's unfinished code: WP1 and WP3 share only the contract (including the locator, §6.6); WP3 reaches WP2's
code only through `run_evaluation_command.py` (runners take the retriever, answer service and settings as arguments, typed with
`TYPE_CHECKING` imports), so each package's tests pass on its own.

### 14.1 WP1: offline index builder and the shared encoders
- **Owns:** `retrieval_index_builder/` (everything) and `gpt2_service_huggingface_space/retrieval_encoders/`.
- **Must not touch:** `gpt2_demo_service/`, `tests/` of the service, `retrieval_contracts/`, the Dockerfile, the evaluation.
- **Folders:** `retrieval_index_builder/{source_alignment, passage_chunking, bundle_writing, tests, build_output}` exactly as §9.
- **Inputs:** the data files of §2 fact 9, the raw books, the contract (`rare_shingle_location` for pair placement).
- **Delivers:** the interfaces of §8.1-8.2 and the rules of §8.3-8.5 exactly; `build_output/bundle/` (real bundle) with
  `build_output/build_report.json` (chapter alignment per book, passage count, token percentiles, pair placement counts by method
  and the chapter rate per book, encoder parity result, timings); two ablation bundles `build_output/ablation_bundles/target_128/`
  (128/150) and `target_256/` (256/290), never uploaded; the upload to `karamAM/gpt2-got-retrieval-index` (private, verified) and
  `build_output/upload_receipt.json` (repo, commit, index version); `build_output/model_files/` with both pinned models.
- **Tests (must pass without network or model files):** conformance: from `fixture_bundle/*.jsonl` records + `HashingTextEncoder`,
  `write_index_bundle` reproduces the fixture's lexical arrays exactly and vectors to 1e-6, and its own manifest passes
  `find_bundle_problems`; `compose_index_version("2026-10-24T00:00:00Z", fixture manifest files) == "got-retrieval-20261024-eeb6e60a"`;
  BM25 weights equal a naive loop on random small corpora and order-agree with `rank_bm25` (top 3); chunker: never crosses a
  chapter, every text is an exact substring, ceiling respected for 150 / 230 / 290, overlap rule; heading regex on synthetic
  split-letter headings ("C ATELYN", "PROLOGU E", "BRIENNE I t was", "CHAPTER ONE ARYA"); `place_pair` on synthetic parts: each of
  the five methods, a window clamped at a part edge, a span centre on a chapter edge, and a pair with a window never placed outside
  it; upload refuses a public repo (mocked `HfApi`). Encoder tests that need the real model (parity, the scorer's input cap) are
  marked `slow` and run once.
- **Done when:** tests green; real bundle built, ≤ 250 MB; report reviewed for the 95 % alignment gate (or the book-only fallback
  stated); pair chapter rate reported (expect about 58 % shingle-placed, §2 fact 14); both ablation bundles built; uploaded privately
  with the receipt written.

### 14.2 WP2: runtime retrieval and API inside the service
- **Owns:** `gpt2_service_huggingface_space/` except `retrieval_contracts/` and `retrieval_encoders/`: `gpt2_demo_service/`, `tests/`,
  `Dockerfile`, `.dockerignore`, `requirements*.txt`, `pytest.ini`, `README.md`, `AGENTS.md`, and deletes the 13 old flat files once
  moved.
- **Must not touch:** the publish script (integrator), the two shared packages, the offline packages, the portfolio.
- **Folders:** `gpt2_demo_service/{language_models, index_loading, index_search, grounded_answers, monitoring, web_api}` + `tests/`
  exactly as §9.
- **Inputs:** the contract and fixture; the encoder protocols (use `HashingTextEncoder` / `TokenOverlapPairScorer` until WP1's
  `retrieval_encoders` lands; the production loader imports `PinnedModelTextEncoder` / `PinnedModelPairScorer` lazily by name).
- **Delivers:** §7 exactly (per-package settings + `service_settings.py`, preallocated KV cache, time limit, layouts, packing, index
  loading and search, grounded answers with `answer_from_evidence`, one-snippet citations, monitoring, web API with the two response
  models, try-it page), §10.1 image files, a rewritten service AGENTS.md and README (retrieval, sources, the one-snippet 25-word
  policy, no mention of where the study came from).
- **Tests:** cached vs full forward logits equal to 1e-5 on a random tiny model, step by step; the cache never reallocates (keys and
  values keep their `data_ptr()` across every decoding step; appending past capacity raises); generator stop rules including
  "time_limit" (a limit of 0 stops after the first token); layouts render byte-exact (question-only = today's string); packer
  property tests over random budgets (never > 1,024 with reserve, question intact, limits respected, only the last item trimmed,
  adjacent chunks allowed, a more-than-half overlap skipped); against the fixture bundle with the fixture settings (candidate and
  rerank counts ≥ 6, evidence floor 0.15, pair threshold 0.45): rankings, fused order, best dense score and pair rankings equal
  `fixture_expectations.json`; fg-5 / fg-6 → `no_evidence`; bundle loader refuses a changed byte, a wrong fingerprint and a wrong
  tokenizer version; citations: exactly one book snippet (the first passage), the same snippet for two different questions, other
  passages `snippet` None, a pair's snippet is its question, ≤ 25 words; guard cuts a copy of the 33-word Ysolde fixture passage;
  `answer_question` equals `answer_from_evidence` for the same evidence and seed; settings: `load_service_settings({})` equals the
  defaults, each env override parses, a bad value raises `ValueError` naming the variable; boundaries: in a fresh subprocess,
  importing `gpt2_demo_service.index_search.evidence_retriever` loads neither torch, pydantic nor fastapi, and no module except
  `web_api/app_factory.py` imports `service_settings`; API: old body → `completion` string and no `text`, spam body exactly
  `{"completion"}`, GoT body keys exactly {completion, retrieval_used, sources} and every source exactly the four keys (including a
  source with no chapter), 422 / 429 / 503 unchanged, `/health` keeps its old keys, `Server-Timing` present; startup with a broken
  bundle still serves `/generate`. A slow, opt-in test compares 20 greedy tokens with and without the cache on the real weights.
- **Done when:** `python -m pytest` green in the service folder (contract self-test included, 11 tests); the §10.1 resolver gate
  exits 0; `docker build` is not available on the Mac, so the integrator's first Space build is the image test; app starts locally
  with fakes via `create_app`.

### 14.3 WP3: evaluation harness
- **Owns:** `retrieval_evaluation/` (everything).
- **Must not touch:** anything else. It imports the contract (including `rare_shingle_location`) and, only in
  `run_evaluation_command.py`, the service's `service_settings`, `index_loading`, `index_search`, `grounded_answers` and
  `language_models` public functions. Never `retrieval_index_builder`.
- **Folders:** `retrieval_evaluation/{golden_sets, metrics, experiments, live_checks, tests, results}` exactly as §9.
- **Inputs:** the contract and fixture; the validation pairs, the cleaned book parts (for the locator and the FD evidence check),
  the fact rules, the local weights; later the real bundle and ablation bundles (WP1) and the real pipeline (WP2).
- **Delivers:** §11 exactly: golden sets (G1 block split, FD, G2, G3, G4), metrics, ablation runner, calibration (floor + leave-one-out
  threshold), answer runs through `answer_from_evidence` with the run-time cap, live checks, report writer; the command
  `python -m retrieval_evaluation.run_evaluation_command {golden|retrieval|calibrate|answers|live|all}`; outputs in `results/`
  (`golden_sets.json`, `retrieval_ablation.json`, `calibration.json`, `answers.json`, `EVALUATION-REPORT.md`,
  `recommended_settings.json` with the fields of §7.1 it decides plus `raft_needed` and the evidence for each). The report states
  the G1 inflation (§2 fact 13) and A5's margin.
- **Tests (fixture only):** metric functions on hand cases (relevance `[0,1,0,1,0]`, 2 relevant: hit@1 0, hit@3 1, precision@3
  1/3, recall@3 1/2, recall@5 1, reciprocal rank 1/2, normalized discounted cumulative gain at 5 0.650921 and at 3 0.386853); hit
  within the packed budget on hand cases; G1 builder on the fixture (question + gold answer through the locator): fg-3 and fg-4 are
  located in part 1 with at least one hit inside their gold sentence, fg-1 and fg-2 have fewer than 2 rare hits and are dropped;
  relevance: a passage holding a hit whole is relevant, one cutting a hit is not; block split on synthetic centres (dev, test and
  edge-dropped cases); FD rules (about 40 questions, none of the 30, token F1 ≤ 0.6 to each G2 question, evidence fields present;
  the exact-match check against the 34,001 pairs and the evidence check against the books run as `data` tests on the Mac); the RRF
  example of the expectations; calibration picks the fixture floor that separates fg-1..4 from fg-5/6; leave-one-out never returns a
  pair as its own neighbour; the copyright checker flags a 26-word copy of a passage that was NOT packed and a response with two
  book snippets; `load_fact_rules` returns 30 rules whose first 10 questions equal the known ten; the answer runner (with spy fakes
  of the §7.5 / §7.6 signatures) retrieves once per question and calls `answer_from_evidence` once per configuration and sample.
- **Done when:** tests green; `golden` built on the real data with counts reported (expect 443 located, 189 dev, 196 test, FD about
  40 checked); once WP1 + WP2 land, the integrator runs `all`.

### 14.4 Shared fixture (exists; read-only)
`retrieval_contracts/shared_fixture/`: two invented books, 6 passages, 4 pairs (one with no book or chapter, standing for a pair no
rule places), 4 chapters, 6 golden questions (two unanswerable), a complete reference bundle built with `HashingTextEncoder` (256
dimensions, instruction "search passages: ", index version `got-retrieval-20261024-eeb6e60a` by the §6.3 rule), and
`fixture_expectations.json` with tokens, BM25 settings, the RRF example, the fixture floor and threshold, and per-question rankings.
Loaders in `fixture_files.py`; 11 self-tests in `test_shared_fixture.py` (2 for the locator). It is the integration contract in
miniature: WP1 must produce it, WP2 must read it the same way, WP3 must score it the same way.

### 14.5 Interfaces between the packages (summary)
| From → to | What crosses | Where it is fixed |
|---|---|---|
| contract → WP1, WP3 | `build_rare_shingle_index`, `locate_text`, `SearchWindow`, `LocatedSpan`, `ShingleHit` | §6.6, `retrieval_contracts/rare_shingle_location.py` |
| WP1 → WP2 | bundle files; `PinnedModelTextEncoder`, `PinnedModelPairScorer(input_token_limit=…)`, `download_model_files` | §6, §8.1, `retrieval_contracts/` |
| WP1 → WP3 | the bundle, the two ablation bundles, the model files folder | §6, §8.2 |
| WP2 → WP3 | `load_service_settings`, `EvidenceSearchSettings`, `GroundedAnswerSettings`, `PromptLayout`, `load_index_bundle`, `EvidenceRetriever.retrieve`, `RetrievedEvidence`, `build_empty_evidence`, `pack_prompt`, `load_got_text_generator`, `GroundedAnswerService.answer_from_evidence` / `answer_question` | §7.1-7.6 |
| WP3 → integrator | `recommended_settings.json`, `EVALUATION-REPORT.md` | §11.6 |

## 15. Risks and open items
- **Prompt-only RAG may not help a 355M model** (research §0, and §2 fact 3: the question block moves to positions never
  fine-tuned): mitigated by four layouts, a fixed RAFT rule, and the option to ship with retrieval off. Expect RAFT.
- **Chapter alignment quality:** the 95 % gate and the book-only fallback keep citations honest.
- **Latency:** the KV cache is the critical path; if A10 fails, lower `context_token_budget` first, then the reranker off; the
  line cap follows the live p95 (D24).
- **Small reported set:** about 196 G1 test questions give A5 a margin of about ±6 points; the report states it.
- **FD questions are written by a builder:** the evidence-offset check guards against wrong facts; Karam may read the 40 lines.
- **Snippet collection:** passage openings (25 words each) could be gathered over many calls; bounded by 20 calls per minute per
  address and the `GOT_BOOK_SNIPPET_ENABLED` switch.
- **Pair answers that differ by time** ("who sits on the Iron Throne"): no timeline logic; the regression set watches it.
- **Space secret is a write-scoped token** (inventory §1.7): a read-only token for the weights + index repos is safer; creating
  it is Karam's click in the Hub settings, not a blocker.
- **Public Space logs:** treated as public; no visitor or book text in them.

## 16. Change log
- 2026-10-24: first version (architect, part 4).
- 2026-10-24: part 6, after the senior review (part 5, 18 fixes): tokenizers 0.23.3 + resolver gate; two response models instead of
  `response_model_exclude_none`; answer time limit and line-cap rule; G1 block split; fact-dev set FD for every choice, G2/G4 run
  once; leave-one-out pair calibration; FD evidence gate A7b and hybrid-by-default; unwidened hit relevance and packed-budget
  comparison of chunk sizes; per-package settings without defaults, `index_loading/` split out, no re-exports; one answer path
  `answer_from_evidence`; one question-independent book snippet per answer, A16 against all passages; packer overlap and trim
  rules, `max_passage_tokens` removed; preallocated KV cache; reranker over the top 10 at ≤ 256 word pieces with a cost prior;
  `index_version` over all 7 file checksums (comment in `bundle_manifest.py`, fixture value re-derived, no format bump); pair
  chapters by rare shingles through the new frozen `retrieval_contracts/rare_shingle_location.py` (+ 2 self-tests, comment updated
  in `bundle_records.py`); folder renames (`web_api/`, `source_alignment/`, `metrics/`, `experiments/`), spelled-out keys,
  `.dockerignore`; facts on trained positions, the 2025 model's view of part 6 and `Server-Timing` pass-through.
- 2026-10-25: part 10 (integrator), outcomes against this design: chunk size 128/150 won the packed-budget rule and is the live bundle (commit `e2925f0e`); prompt-only retrieval failed the facts rule, so the §12 grounding fine-tune ran (19,289 sequences with passages only, no pair examples, 68 GPU minutes) and is live; the Mac-time x 3 planning factor was replaced by the live-measured ratio (1.0) and the reranker was switched on with its own evidence floor 0.102; the 25-word snippet now counts words like the A16 check; line cap 5; A1 (turned wrong), A3, A7, A8 not met. Details: `../docs/RESULTS.md`.

## 17. Review answers (part 5 points not applied as written)
Fixes 1, 2, 3, 6, 7, 8, 10, 12, 13, 14 and 18 are applied as written. The rest, one line each:
- Fix 3: applied, plus the env override `GOT_MAX_REQUESTS_IN_LINE`, so step 6 sets the cap without a code publish; default cap 3.
- Fix 4: applied; the old "expect ≥ 500 kept" was wrong: the locator measures 443 located, 189 dev and 196 test after the edge drop.
- Fix 5: applied; FD questions also carry an evidence offset checked against the books, because a builder writes them.
- Fix 9: no `language_models/generation_settings.py`: the generator needs one value (the time limit) as a constructor argument, and
  a ninth module would push that folder past the 8-file limit; the model-loading values live in `service_settings.ModelSettings`.
- Fix 9: `grounded_answers/prompt_settings.py: PromptSettings` became `answer_settings.py: GroundedAnswerSettings`, because it also
  holds the answer token floor and ceiling and the snippet switch (the old name would lie).
- Fix 9: settings are plain frozen dataclasses everywhere (pydantic only for the API schemas), so no settings object needs pydantic.
- Fix 11: applied, plus the switch `GOT_BOOK_SNIPPET_ENABLED` to turn book snippets off without a deploy.
- Fix 15: applied, and the fixture manifest's `index_version` string was re-derived (`…-eeb6e60a`) so the fixture follows its own
  rule; `manifest.json` carries no checksum, and the 11 self-tests pass.
- Fix 16: the locator lives in the frozen contract, not in the builder: WP3 builds in parallel with WP1 and must not import WP1's
  unfinished code. Pairs with a coarse window are searched only inside it (never corpus-wide), so a coarse location is never
  contradicted; measured 98 % agreement on 2,001 pairs.
- Fix 17: applied; also `index_search/` would have held 9 modules with its settings file, so download and loading moved to a new
  `index_loading/` package.

## 17. Changes after the first release (2026-10-25, hardening pass of `wf-1024-2000-got-rag`)
The sections above are the design as built for the first release; where they differ from the list below, this list wins.
- Copy guard (sections 6.4 and 10): `grounded_answers/book_run_index.py` holds one 64-bit hash for every run of 26 consecutive words across ALL
  26,112 passages (neighbouring passages that share a sentence are joined first, so a run may cross a passage edge), about 14 MB, built at
  start-up in about 2 s; `verbatim_guard.limit_verbatim_overlap(completion, book_run_index)` cuts at the first hit on every route.
  `GuardedCompletion.longest_shared_run` and `MAX_SHARED_WORDS` are gone; the evaluation's independent check stays independent.
- `QuestionTooLongError` (raised by `answer_question` before any search) becomes a 422 "Questions can be up to N tokens."; the answer reserve is
  also capped so prompt + reserve never exceed the context.
- Pair search: the magic `pair_example_threshold=2.0` is replaced by the flag `EvidenceSearchSettings.pair_examples_enabled` (default False,
  env `GOT_PAIR_EXAMPLES_ENABLED`); the search is skipped when the flag is off or the layout shows no pairs, and a pair layout without the
  flag stops the start with a message.
- Request defaults (temperature, top-p, top-k, repetition penalty, answer length) live in `service_settings.RequestDefaults`; the request fields
  are optional and the server fills the gaps.
- `web_api/retrieval_startup.py` moved to the new package `service_startup/` (web_api would hold 10 files), it also builds the copy guard's index.
- Visitor limit: the visitor address is the rightmost public entry of `X-Forwarded-For`; refused callers are not stored.
- Builder: `source_alignment/heading_reading/` holds the three raw-heading readers; the text after the last chapter is titled "Appendix"
  (chunker version 2, effective at the next index rebuild).
