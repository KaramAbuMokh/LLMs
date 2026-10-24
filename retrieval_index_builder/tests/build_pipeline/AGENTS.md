# tests/build_pipeline — AGENTS.md
Purpose: tests of the glue around the stages: the command's argument rules, the build report, loading the part texts, and one small whole build on invented books.

## Files
- `test_build_command_arguments.py` — defaults for the real bundle, ablation sizes and folder, and the refusal to upload an ablation bundle.
- `test_build_report_writing.py` — passage percentiles, chapter rates per book, and that the written report has every section and no book text.
- `test_book_part_texts.py` — seven parts with the held-out stretch as part 6, a missing training part is refused.
- `test_small_end_to_end_build.py` — invented dataset through alignment, pair placement, chunking and bundle writing with the hashing encoder; the bundle passes its own check.
