# retrieval_index_builder — AGENTS.md
Purpose: the OFFLINE half of the Game of Thrones retrieval work (workflow `wf-1024-2000-got-rag`, design in `../docs/DESIGN.md` section 8). It reads the cleaned
book text, the raw books (only for chapter headings) and the 34,001 cleaned question-answer pairs, and writes ONE retrieval bundle: passages with chapters, pairs with books
and chapters, dense vectors, lexical (BM25) postings and a manifest with checksums. The bundle is uploaded only to the PRIVATE Hugging Face dataset repo
`karamAM/gpt2-got-retrieval-index` (the books are copyrighted). It never imports `gpt2_demo_service`; it shares only the frozen contract `../gpt2_service_huggingface_space/retrieval_contracts/`
and the encoders `../gpt2_service_huggingface_space/retrieval_encoders/` (this package's own work, shipped in the Space image).

## How to run (from this project folder, venv `~/.venvs/got-retrieval-index-builder`, Python 3.11)
- Tests: `cd retrieval_index_builder && python -m pytest` (the slow real-model tests: `python -m pytest -m slow`; they need the files in `build_output/model_files/`).
- Models: `PYTHONPATH=gpt2_service_huggingface_space python -m retrieval_encoders.model_file_download --target retrieval_index_builder/build_output/model_files`.
- Build: `PYTHONPATH=gpt2_service_huggingface_space:. python -m retrieval_index_builder.build_index_command [--upload]` writes `build_output/bundle/` and `build_output/build_report.json`.
- Ablation bundles (never uploaded): add `--target-tokens 128 --max-tokens 150 --output-folder retrieval_index_builder/build_output/ablation_bundles/target_128` (and 256 / 290 for `target_256`).
- Encoder parity (needs sentence-transformers, so a scratch venv only, not a dependency): `python -m retrieval_index_builder.encoder_parity_check` writes `build_output/encoder_parity.json`, which the next build report includes.
- The upload reads the token file named in the settings at run time and never prints or stores it.

## Files
- `index_build_settings.py` — every setting of a build in one frozen dataclass (data and output folders, passage sizes 200/230 tokens, overlap, tail size, the 20,000-character pair window, repo id, batch size, the 95% chapter gate) and the default settings builder.
- `build_index_command.py` — the command line: loads the texts, aligns chapters, places the pairs, cuts passages, embeds and writes the bundle, writes the report, optionally uploads (default bundle only) and writes the upload receipt.
- `build_report_writing.py` — turns one build into `build_report.json`: alignment per book, passage count and token percentiles, pair placement counts by method, chapter rates per book, encoder parity, stage timings. No book text.
- `encoder_parity_check.py` — compares our ONNX encoders with the sentence-transformers reference on invented sample texts (cosine of the vectors, order of the reranker scores).
- `pytest.ini` — test settings (import paths to the contract and encoders, the `slow` marker is off by default).
- `requirements.txt` — pinned packages for the builder venv.

## Subfolders
- `source_alignment/` — where text, chapters and pairs sit in the books: part texts, raw headings, chapter alignment, pair placement, pair records.
- `passage_chunking/` — sentence splitting, GPT-2 token counting, passage planning and passage records.
- `bundle_writing/` — lexical postings, vector files, bundle assembly with the manifest, builder checksum, private upload.
- `tests/` — unit tests per area, all offline except the `slow` ones.
- `build_output/` — generated, private, never committed (bundles, report, receipt, downloaded model files).
