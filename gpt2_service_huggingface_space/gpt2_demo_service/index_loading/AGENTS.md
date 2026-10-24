# index_loading — AGENTS.md
Purpose: gets the retrieval bundle from the PRIVATE dataset repo and turns it into read-only arrays in memory. Needs numpy and huggingface_hub, no torch,
no pydantic. A bundle that fails any check is refused with reasons that name files and numbers, never text.

## Files
- `bundle_download.py` — downloads the manifest and the seven data files at one full commit hash (a branch name or short hash is refused before any network call).
- `bundle_loading.py` — checks the manifest (file checksums, lexical tokenizer version, indexed-text version, encoder fingerprint, record and vector counts), then loads the records, the vector matrices and the lexical postings into frozen records; `BundleValidationError` carries the reasons. `load_lexical_postings` also takes the document count, because the stored arrays do not hold it.
