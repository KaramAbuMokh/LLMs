# Test setup shared by every builder test: keep the GPT-2 vocabulary offline and give tests the real settings' data folder name
import os  # environment for the tokenizer cache
from retrieval_index_builder.index_build_settings import PROJECT_FOLDER  # project location

TIKTOKEN_CACHE_FOLDER = PROJECT_FOLDER / "kaggle_finetune" / "mixed_run" / "staging" / "mixed_books_qa_data"  # holds the two GPT-2 vocabulary files named by hash
if TIKTOKEN_CACHE_FOLDER.is_dir():  # the files sit next to the book data on this Mac
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE_FOLDER))  # tiktoken then never needs the network
