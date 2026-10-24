---
title: GPT-2 Demos
sdk: docker
app_port: 7860
pinned: false
short_description: Fine-tuned GPT-2 that talks like Game of Thrones
---

# GPT-2 demos

Two small fine-tuned GPT-2 models behind one service, built by Karam Abu Mokh for his portfolio (karamam.com).

- **Game of Thrones (GPT-2 medium, 355M parameters)** fine-tuned on the five books and thousands of question-answer pairs about them. It answers in the voice of the books. It learned the style better than the facts, so some details are wrong.
- **Spam check (GPT-2 small)** fine-tuned to answer `spam` or `not spam`. It was trained on short SMS-style texts, so it is best on those.

The weights live in a private model repo and are downloaded at startup. The Space runs on free CPU, so after 48 hours without visitors it sleeps and the first request has to wake it, which takes about a minute.

## Looking things up before answering

Before the Game of Thrones model answers, the service searches a private index of the books (passages of about 130 tokens, never crossing a chapter) and of solved question-answer examples. It combines a word search and a meaning search, re-reads the best ten with a small cross-encoder, keeps only evidence that is close enough, and puts up to three book passages in front of the question in the exact format the model was trained on. The model was fine-tuned a second time on questions that come with their passage, so it learned to answer from the passage in front of it instead of guessing. The solved examples are indexed but not shown to the model: the evaluation found no similarity level at which a neighbour's answer was reliable enough. When nothing close enough is found, the model answers as it always did and `retrieval_used` is `false`.

The books are copyrighted, so the index is not public. Answers name their evidence: the book and chapter of each source, plus ONE short snippet of the book (at most 25 words). The question chooses WHICH passage is cited; the snippet itself is always the opening of that passage, so a question cannot aim it at a line. Every answer, with or without a lookup, is also checked against every passage of the books: if it repeats more than 25 consecutive words of any book text, it is cut back.

## API

- `GET /` is a try-it page: ask the Game of Thrones model a question in the browser; the sources appear under the answer.
- `GET /health` shows that the service is up, which models are loaded, the memory use and whether retrieval is ready (index version, counts, reason if it is off).
- `POST /generate` takes JSON `{"prompt": "...", "task_name": "got" | "spam", "max_new_tokens": 120}` (optional `use_retrieval`: `false` skips the lookup).
  - Game of Thrones returns `{"completion": "...", "retrieval_used": true, "sources": [{"source_type": "book_passage", "book": "...", "chapter": "...", "snippet": "..."}]}`; every source always has the four keys, and `book`, `chapter` and `snippet` can be `null`. The response also carries a `Server-Timing` header with the time spent on retrieval, reading the prompt and writing the answer.
  - Spam returns exactly `{"completion": "spam"}` or `{"completion": "not spam"}`.

Limits that keep a public CPU Space safe: questions up to 500 characters (emails up to 2,000), answers up to 200 tokens and at most 25 seconds of writing, one generation at a time with a short line (`503` when full) and at most 20 requests per minute per visitor (`429`, counted by the visitor's address as the Hugging Face proxy reports it). A question that is too long for the model (more than 864 tokens, which 500 characters of emoji reach) gets `422`.

Logs hold no visitor text and no book text: only ids, scores, timings and token counts.
