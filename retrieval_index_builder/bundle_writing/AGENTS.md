# bundle_writing — AGENTS.md
Purpose: writes the retrieval bundle in the frozen format of `retrieval_contracts` (DESIGN.md 6.3, 8.2) and uploads it to the private index repo.

## Files
- `lexical_postings_writing.py` — builds the lexical postings: sorted terms, posting starts, posting rows and the full precomputed BM25 weight of every (term, document), with the contract's word splitter.
- `vector_writing.py` — encodes texts in batches with any encoder of the contract's interface and saves vectors and postings as plain arrays (no pickles).
- `builder_source_checksum.py` — one checksum over the builder's and the encoders' own source files, stored in the manifest.
- `bundle_assembly.py` — writes the seven data files, then the manifest (last), derives the index version from the file checksums, and refuses a folder that fails its own check.
- `bundle_upload.py` — creates the dataset repo private if missing, REFUSES a public repo, uploads only the eight bundle files in one commit and returns the receipt (repo, commit hash = index revision, index version).
