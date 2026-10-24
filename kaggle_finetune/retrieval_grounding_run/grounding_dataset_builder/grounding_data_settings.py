# Every number and path of the grounding dataset build (DESIGN.md section 12): which pairs, how many passages, which seeds
from dataclasses import dataclass  # frozen settings record
from pathlib import Path  # folder settings

RUN_FOLDER = Path(__file__).resolve().parents[1]  # kaggle_finetune/retrieval_grounding_run/
PROJECT_FOLDER = RUN_FOLDER.parents[1]  # the repository root


@dataclass(frozen=True)
class GroundingDataSettings:  # what one dataset build reads and how it cuts the examples
    source_data_folder: Path = PROJECT_FOLDER / "kaggle_finetune/mixed_run/staging/mixed_books_qa_data"  # pairs, validation pairs, book parts
    bundle_folder: Path = PROJECT_FOLDER / "retrieval_index_builder/build_output/ablation_bundles/target_128"  # the bundle the Space serves (128-token passages)
    model_files_folder: Path = PROJECT_FOLDER / "retrieval_index_builder/build_output/model_files"  # the two pinned encoder models
    golden_sets_path: Path = PROJECT_FOLDER / "retrieval_evaluation/results/golden_sets.json"  # held-out development questions
    output_folder: Path = RUN_FOLDER / "staging/grounding_data"  # the private Kaggle dataset payload
    pair_window_characters: int = 20000  # half-width of a pair's coarse location window (as the index builder placed pairs)
    minimum_oracle_hits: int = 2  # an oracle passage holds at least this many of the pair's rare shingle hits whole
    oracle_share: float = 0.8  # share of examples that include the oracle passage (the rest hold distractors only)
    oracle_distractor_count: int = 2  # hard distractors next to the oracle passage
    distractor_only_count: int = 3  # distractors in an example without the oracle passage
    held_out_part_number: int = 6  # never an oracle or a distractor: the evaluation's own stretch of A Feast for Crows
    oracle_validation_count: int = 300  # pairs kept out of training to score how well the model uses the oracle passage
    context_token_budget: int = 600  # the budget the Space packs with
    answer_reserve_tokens: int = 120  # the Space reserves at least this many tokens for the answer
    safety_margin_tokens: int = 8  # the Space's safety margin
    maximum_answer_tokens: int = 142  # longest answer kept, as in the Q&A training data
    embedding_thread_count: int = 8  # onnxruntime threads on the Mac
    random_seed: int = 20261024  # one seed for every shuffle and every coin flip
