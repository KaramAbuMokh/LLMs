# Every setting of the offline index build in one place (paths, chunk sizes, upload target)
import os  # the token file location from the environment
from dataclasses import dataclass  # frozen settings
from pathlib import Path  # folder settings

BUILDER_FOLDER = Path(__file__).resolve().parent  # retrieval_index_builder/
PROJECT_FOLDER = BUILDER_FOLDER.parent  # the repository root
INDEX_REPOSITORY_ID = "karamAM/gpt2-got-retrieval-index"  # PRIVATE dataset repo that holds the bundle (D14)


@dataclass(frozen=True)  # immutable settings
class IndexBuildSettings:  # what one build reads, how it cuts passages and where it writes
    source_data_folder: Path  # mixed_books_qa_data: the 34,001 pairs, the six book parts, the held-out passage
    raw_books_folder: Path  # the five raw book files (only place with line breaks and contents lists)
    output_folder: Path  # where the bundle is written
    model_files_folder: Path  # downloaded encoder models
    target_tokens: int = 200  # GPT-2 tokens a passage aims for
    max_tokens: int = 230  # hard ceiling per passage
    overlap_sentences: int = 1  # sentences repeated at the start of the next passage of a chapter
    minimum_tail_tokens: int = 40  # a shorter last passage merges into the one before it
    maximum_overlap_tokens: int = 60  # a longer overlap is not repeated
    pair_window_characters: int = 20000  # half-width of a pair's coarse location window (the locations are 20,000-character bins)
    index_repo_id: str = INDEX_REPOSITORY_ID  # upload target
    embedding_batch_size: int = 64  # texts per encoder call while building vectors
    encoder_thread_count: int = 8  # onnxruntime threads for the offline build (the Space uses 2)
    minimum_aligned_share: float = 0.95  # a book with fewer aligned chapter headings ships without chapter fields (DESIGN.md 8.3)
    token_file: Path = Path(os.environ.get("HF_TOKEN_FILE", Path.home() / ".cache" / "huggingface" / "token"))  # read only at upload time, never printed or copied; set HF_TOKEN_FILE or run `huggingface-cli login`


def build_default_settings(output_folder: Path | None = None) -> IndexBuildSettings:  # the settings of the shipped build
    build_output_folder = BUILDER_FOLDER / "build_output"  # everything generated lives here, never committed
    return IndexBuildSettings(  # frozen settings
        source_data_folder=PROJECT_FOLDER / "kaggle_finetune" / "mixed_run" / "staging" / "mixed_books_qa_data",  # pairs and book parts
        raw_books_folder=Path.home() / "Downloads" / "drive-download-20250818T222213Z-1-001",  # raw books with line breaks
        output_folder=output_folder or build_output_folder / "bundle",  # the real bundle unless an ablation overrides it
        model_files_folder=build_output_folder / "model_files",  # downloaded encoders
    )
