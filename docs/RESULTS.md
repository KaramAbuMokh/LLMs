# Game of Thrones model with retrieval: results (2026-10-24 night)

Workflow `wf-1024-2000-got-rag`, DECISIONS `hq.gpt2_got_rag_1024`. This is the plain write-up of what was built, what it did to the answers, what is live now and how to undo it.

## In one paragraph

The model sounds like the books but guessed facts. Now, before it answers, the Space searches a private index of the five books (26,112 short passages), reads the best ten again with a small ranking model, and puts up to three passages in front of the question. Just pasting the passages did NOT help the model (it ignored them), so it was trained a second time, on free Kaggle GPU, on 19,289 questions that come with their passage. With that training the model answers from the passage. It is live on https://karamam-gpt2-demos.hf.space and the portfolio contract (`completion`) still works. The facts score went from 12 to 17 of 30, the new questions from 2 to 13 of 42, and invented names in answers fell from 36% to 12%. Four of the sixteen planned targets are not met; they are listed below, not hidden.

## Before and after (same questions, same seeds, same machine)

"Right" means 2 of 3 samples contain the accepted words and none of the rejected ones (the same rule that gave 10 of 30 on Kaggle). All arms ran on the Mac with the model files that are, or will be, live.

| measure | live model alone (before) | live model + passages pasted in | grounded model + passages (now live) | target |
|---|---|---|---|---|
| 30 fact questions right | 12 | 10 | **17** | gate 15, target 20 |
| same, without the 3 questions that are stored pairs | 10 of 27 | 8 | **14** | gate 12, target 17 |
| the first ten ("known ten") | 5 | 5 | 5 | gate 6, target 8 |
| 42 NEW fact questions (never trained on) | 2 | 8 | **13** | no gate |
| samples right out of 90 | 36 | 31 | **47** | |
| the four old mistakes (Iron Throne x2, Kingslayer, Theon's home) | 1 of 4 | 1 of 4 | **4 of 4** | target 4 |
| Iron Throne answers that name Gregor, Clegane or the Mountain | none | none | none | none |
| 196 held-out passage questions: share of the key words found | 0.052 | 0.078 | **0.213** | gate: not worse |
| same: word overlap with the reference answer (F1) | 0.350 | 0.337 | **0.432** | |
| same: answers that name a person found nowhere in question or passage | 36% | 28% | **12%** | |
| answers that end by themselves (not cut) | 298 of 298 | 293 of 298 | 298 of 298 | |
| copyright findings (a snippet over 25 words, or 26 words copied from any passage) | 0 | 48 | **0** (after a fix, see below) | 0 |

The held-out passage questions are easier for every arm than real visitor questions, because the 2025 starting model read that passage; the gain is still real, the absolute level is flattering. Some answers still go wrong in the old way (a live check answered "Hodor is a direwolf" before the training; the new model gets such questions right more often, not always).

Three of the 12 questions the old model got right turned wrong (Jon Snow's direwolf, the Lord Commander when Jon joins, what Ser Gregor is called); eight others turned right (seat of Stark, words of Lannister, sigil of Baratheon, Robb's direwolf, Daenerys's husband, capital of the Seven Kingdoms, the Kingslayer, Theon's home). With 3 samples per question one question is noise, so "17 against 12" is the honest size of the gain, not a precise number.

## Finding the right passage

| measure (196 test questions, 95% margin about plus or minus 6 points) | words search + meaning search | plus the ranking model (live) |
|---|---|---|
| right passage first | 0.495 | 0.658 |
| right passage in the top 3 | 0.633 | 0.765 |
| right passage in the top 5 | 0.714 | **0.781** |
| mean reciprocal rank at 10 | 0.588 | 0.715 |

Passage size was chosen by the numbers: the right passage within the 600-token budget was found for 66% of the questions with 128-token passages, 63% with 200-token ones, 57% with 256-token ones. So the live index holds 128-token passages (26,112 of them, never crossing a chapter) and the 34,001 question-answer pairs. The test questions were run twice: once without the ranking model (before its cost was known), once with it (final). Both runs are shown.

## What is live now

| thing | value |
|---|---|
| Space | karamAM/gpt2-demos, commit `c304d51d` (hardened; the first retrieval commit `2fe874e6` is the rollback point), public as before, cpu-basic |
| model file | `got_model_retrieval_grounded.pth` (weights repo commit `51cb85f8`, SHA-256 starts 696a1080); the old `got_model.pth` is untouched |
| index | private dataset repo `karamAM/gpt2-got-retrieval-index`, commit `e2925f0e`, version `got-retrieval-20261024-241316fb`; anonymous access is refused (401) |
| search | words + meaning search, top 10 re-ranked by a small cross-encoder, evidence kept when the rank score is at least 0.102, up to 3 passages, 600 retrieved tokens at most |
| prompt | passages, then the exact trained question template (the "qusetion" typo unchanged) |
| not used | stored question-answer examples (no similarity level made a neighbour's answer reliable: best precision 0.47 at similarity 0.95) |
| public answer | `completion`, `retrieval_used`, and per source book + chapter; ONE snippet of at most 25 words from the best passage; every answer on every route is cut at the first stretch of more than 25 words shared with ANY of the 26,112 passages |
| server default temperature | 0.3 (was 0.7; the portfolio sends none, so its contract is unchanged) |
| line cap | 5 requests (the largest cap with cap x slowest measured p95, 10.1 s, under 60 s) |

## Live measurements (no token, as a visitor)

| check | result |
|---|---|
| `/health` | retrieval ready, right index version, reranker on, no problem |
| portfolio body `{prompt, task_name:"got"}` | 200, string `completion`, no `text` key, keys exactly completion / retrieval_used / sources, each source exactly 4 keys |
| spam | exact body `{"completion"}`, real SMS spam is labelled spam, p95 0.7 to 1.1 s (limit 1.5) |
| 30 fact questions, one call each | 18 right (one sample each; the Mac run gives 17 by majority of 3), 0 failed calls, 0 copyright findings against all 26,112 passages |
| speed, one caller, warm | total p50 3.8 s and p95 4.5 s in a fast hour, p95 10.1 s in a slow hour (the free CPU varies about 2x); limit 22 s |
| retrieval stages p95 | 342 ms (fast) to 644 ms (slow) with the ranking model; limit 800 ms. The `Server-Timing` header reaches the visitor |
| plain path | 0.17 s per generated token (limit 0.45); before the cache it was 0.5 to 1.05 s |
| memory | 3.4 GB steady, 3.8 GB peak of 16 GB (limit 4.5 and 6) |
| restart | about 80 s from upload to ready (image layers cached) |

## Targets: what passed and what did not

Passed: A4 (no Gregor on the throne; all four old mistakes right), A5 (passage in the top 5: 0.78, gate 0.70), A7b (97 to 100% of the new fact questions keep evidence), A9, A10, A11, A12, A13, A14, A15, A16 (0 findings), and the answer-quality gate (key words found: 0.213 against 0.052).

Not met:
- **A1, second half:** 3 of the 12 old right answers turned wrong (limit 2). The count itself passes (17, gate 15) but misses the target of 20.
- **A3, the known ten:** 5 of 10, gate 6, target 8. The ten are the hardest ones and the old model already had 5 in this run.
- **A7, off-topic questions:** 35% to 38% of the 40 unanswerable questions still get passages (limit 15%). No setting reaches 15% while keeping 85% of the real questions: half of those 40 sound like the series, and a search cannot tell. A model that gets irrelevant passages answers about as badly as before.
- **A8, stored examples:** no similarity level reached the precision target, so no stored example is ever shown (the code path stays, switched off by the explicit flag `pair_examples_enabled`, which since the hardening pass also skips the pair search).
- A2 target 17 (14 reached, gate 12 passed).

## What was decided on the way, and why

- **Prompt-only retrieval was tried first and rejected by the rule:** with the old weights the facts score fell (12 to 10), so the grounding fine-tune was needed (all three triggers fired). The fine-tune ran once on Kaggle (free T4, 68 minutes, stopped early at step 6,400 of 6,848, best at 5,200): loss on the answer with retrieved passages 1.65 to 1.39, with the answer passage 1.14 to 0.74, no evidence 2.04 to 2.08 (about the same), the books' voice 2.56 to 2.57. It starts from the live mixed model, still ends its answers by itself (10 of 10 on the plain check), and its training sequences never contain the held-out stretch.
- **Ranking model on:** the Mac rule (Mac time x 3) said no. The live test showed the real cost, 265 to 644 ms on the Space, so the factor 3 was replaced by 1 and the whole retrieval and calibration stage was re-run into its own folder; the rule then said yes (+11.6 points of top-3). The main results folder still holds the first run.
- **A bug found by the evaluation and fixed:** the 25-word snippet counted words between spaces, so a passage with dashes ("word--word") showed 27 words by the independent count. Words are now counted the same way as in the check (letters and digits). 93 of the 613 retrieval answers of the first evaluation run were affected; 0 afterwards (live and Mac).
- **Index bundle swap:** the 200-token bundle was uploaded first (commit `c7028f40`); the evaluation chose 128-token passages, so that bundle was uploaded as `e2925f0e`, downloaded again, checksummed and found private.

## Hardening pass (2026-10-25, after the senior review; Space commit `c304d51d`, first hardened commit `c6899d39`)

Four things were proven wrong on the live Space and fixed, plus six smaller ones. Tests after the pass: service 313 passed (1 slow test skipped), evaluation 77, index builder 79, grounding run 9.

| problem | proof (live, 2026-10-25) | fix | check after the publish |
|---|---|---|---|
| the per-visitor limit could be bypassed | 20 spam calls with `X-Forwarded-For: 198.51.100.77` gave 200, the 21st 429, then a new fake address gave 200 | the visitor is the rightmost PUBLIC entry of the header (the edge writes the real caller there; verified with a throwaway echo Space), else the socket; refused callers are no longer stored | 20 calls 200, the 21st 429, a NEW fake address 429, a call without the header 429 |
| a long emoji question crashed | 500 emoji = 1,533 tokens gave HTTP 500 | the question block is counted before any search; over 864 tokens gives 422 "Questions can be up to 864 tokens." and the place in line comes back | live 422 with that text |
| log privacy | the memory line carried the caller's task name | fixed words `after_got` / `after_spam`; task names are at most 32 characters | tests (no live read of the log) |
| empty stop text | `""` matches at once and blanks the answer | each stop text has at least 1 character | live 422 |
| copyright guard saw only 3 passages | a memorised book stretch in an answer without evidence was never cut | one hash for every run of 26 words of all passages (1,767,570 runs, 14 MB, about 2 s at start; neighbouring passages are joined, so a run may cross a passage edge); every route is cut at the first hit | 3,000 random real stretches: every one of 26 or more words was cut (2,384 cuts), none of 25 or fewer, the independent check found 0 problems in the cut texts; 240 of those stretches cross a passage edge and a per-passage index would have missed them; live 30 facts: 0 copyright findings |
| pair search ran on every request | magic threshold 2.0 | explicit flag `pair_examples_enabled` (off); the search is skipped; a pair layout without the flag stops the start with a clear message | `/health` unchanged, retrieval stage times lower |
| answers flip from run to run | single samples changed on 2 of 10 questions | default temperature 0.3 | see below |

Temperature choice (Mac, live model, retrieval on, same seeds, three samples per question; tuned on the 42 fact-development questions, confirmed once on the 30 facts):

| temperature | fact-development right | samples right of 126 | 30 facts right | samples right of 90 | questions with mixed samples (30 facts) | 4 old mistakes |
|---|---|---|---|---|---|---|
| 0.3 (now live) | **14** | **44** | 17 | **51** | **6** | 4 of 4 |
| 0.5 | 13 | 41 | not run | not run | not run | not run |
| 0.7 (before) | 13 | 41 | 17 | 47 | 10 | 4 of 4 |

0.3 is not worse anywhere and steadier; the gain in questions is small (one question), so read it as "less random, never worse", not as a big jump. The measure "answers naming a person found nowhere in question or passages" was 24 against 18 answers on the 30 facts and 36 against 36 on fact-development, so it did not clearly move (it also counts right facts that are not in the passages).
Same comparison live on the Space (30 facts, three passes each, temperature sent explicitly): 0.3 gave 18, 14 and 17 right (49 of 90 samples, 18 right by majority of three, 5 questions with mixed samples), 0.7 gave 16, 13 and 14 (43 of 90, 15 by majority, 6 mixed). The Mac run and the live run agree on the direction; the live run also shows the size of the per-pass noise (plus or minus 2 to 3 questions) that one single sample hides.

Live check after the final publish (no token, as a visitor): contract problems none; 30 facts one call each 19 right (before the pass: 18; one sample is noisy, plus or minus 2), known ten 7, 0 failed calls, 0 copyright findings against all 26,112 passages; total p50 3.3 s and p95 4.3 s; retrieval stages p95 311 ms; spam body exactly `{"completion"}`, p95 0.65 s; memory 3.5 GB steady, 3.77 GB peak; the fake-address probe: 20 calls 200, the 21st 429, a new fake address 429 (before: 200); the emoji question 422 (before: 500).

Other changes: the copy-guard index and the pair flag are covered by tests; the web_api folder lost `retrieval_startup.py` (now `service_startup/`); the index builder's `source_alignment/` lost its three heading readers (now `heading_reading/`) and will title the text after the last chapter "Appendix" at the next rebuild (650 passages, 2.5 percent, still cite no chapter in the live index); trailing comments were added to the files the review named. Rollback of this pass: `publish_gpt2_space.py --rollback-folder space_rollback_snapshots/before_hardening_2fe874e6` (brings back the spoofable limit and the 500 on long questions).

## Undo it

| level | how | time |
|---|---|---|
| retrieval off, same model | Space variable `GOT_RETRIEVAL_ENABLED=false` (`set_gpt2_space_variable.py`) | restart about 35 s |
| ranking model off | `GOT_RETRIEVAL_RERANKER_ENABLED=false` | restart |
| old model back | `GOT_WEIGHTS_FILE_NAME=got_model.pth` (the books + Q&A mixed model) | restart |
| old code back | `publish_gpt2_space.py --rollback-folder space_rollback_snapshots/before_retrieval_3103589f` (the Space as it ran at commit `3103589f`) | rebuild, a few minutes |

## Open items for Karam

- Nothing blocks. The Space secret `HF_TOKEN` is still a write-scoped token; a read-only token for the two private repos would be safer (a click in the Hub settings).
- To push the facts higher: a second fine-tune pass with more oracle questions for the weakest kinds (the known ten), or answers written by a stronger model for the ten. Not started.
- The portfolio page needs no change (the new fields are optional). Showing "From the books: book, chapter" there is a small later step on a branch.

## Where things are

- Service code: `gpt2_service_huggingface_space/` (design: `docs/DESIGN.md`).
- Index builder and its private outputs: `retrieval_index_builder/`.
- Evaluation, all result files, the report of the first (prompt-only) run: `retrieval_evaluation/` and `retrieval_evaluation/results/`.
- Integration scripts and results (start-up check, reranker variant, grounded-model scores, live verification): `retrieval_integration/`.
- Grounding fine-tune (dataset build, Kaggle code, results): `kaggle_finetune/retrieval_grounding_run/` and `kaggle_finetune/retrieval_grounding_results/`.
- Publish and rollback tools: on Karam's Mac, not in this repository.
