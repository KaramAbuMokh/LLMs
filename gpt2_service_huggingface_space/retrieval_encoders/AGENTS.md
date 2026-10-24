# retrieval_encoders — AGENTS.md
Purpose: the two small public encoders used by the retrieval work, run with ONNX Runtime on CPU: the dense text encoder (bge-small-en-v1.5, 384 dimensions) that turns book passages and
visitor questions into unit vectors, and the optional cross-encoder reranker (ms-marco MiniLM) that scores a question and a passage together. The SAME classes run offline when the index is
built and online in the Space, and both satisfy the interfaces of `../retrieval_contracts/encoder_contracts.py`. Both models are pinned to exact commits (checked in the Docker build and by the
ONNX file checksum in the bundle manifest). Owner: the index builder work package; tests live in `../../retrieval_index_builder/tests/encoders/`. Design: `../../docs/DESIGN.md` 8.1.
Depends on numpy, onnxruntime, tokenizers and huggingface_hub; does not import torch or the service.

## Files
- `model_file_download.py` — folder naming per model, download of the ONNX graph and tokenizer at the pinned commit, and the command `python -m retrieval_encoders.model_file_download --target <folder>` the Dockerfile runs.
- `onnx_session_building.py` — builds the CPU inference session (given thread count, one operator at a time).
- `word_piece_tokenizing.py` — loads the tokenizer with a length limit (never above what the model reads) and pads a batch into the integer inputs the graph declares.
- `pinned_text_encoder.py` — the dense encoder: texts cut at 512 word pieces, first-token (CLS) vector, unit length, rows returned in the caller's order; reports its fingerprint (model, commit, ONNX checksum, width, pooling, search instruction).
- `pinned_pair_scorer.py` — the reranker: each (question, passage) pair cut at a given word-piece limit (256 in production), one raw score per passage.
