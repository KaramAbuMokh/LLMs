# Test setup: the GPT-2 tokenizer files come from the mixed dataset folder, so no test needs the network
import os  # environment variable
from pathlib import Path  # dataset folder

os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(__file__).resolve().parents[2] / "mixed_run/staging/mixed_books_qa_data"))  # tiktoken cache files sit there
