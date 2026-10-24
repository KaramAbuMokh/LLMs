# The ONE place for every path and number of the evaluation (DESIGN.md §11); tests build their own small copies with dataclasses.replace
from dataclasses import dataclass, field  # frozen settings groups
from pathlib import Path  # data and result locations
from retrieval_contracts.book_parts import HELD_OUT_PART_NUMBER  # the held-out stretch of A Feast for Crows

PROJECT_FOLDER = Path(__file__).resolve().parents[1]  # the repository root
KAGGLE_FOLDER = PROJECT_FOLDER / "kaggle_finetune"  # training data and reports
BUILD_OUTPUT_FOLDER = PROJECT_FOLDER / "retrieval_index_builder" / "build_output"  # where the index builder leaves its bundles


@dataclass(frozen=True)
class DataLocations:  # where the evaluation reads and writes
    source_data_folder: Path = KAGGLE_FOLDER / "mixed_run/staging/mixed_books_qa_data"  # pairs, validation pairs, book parts
    fact_rules_path: Path = KAGGLE_FOLDER / "mixed_run/mixed_kernel_source/fact_questions.py"  # the 30 questions and is_right
    run_settings_path: Path = KAGGLE_FOLDER / "kernel_source/run_settings.py"  # the known ten, imported by the fact rules
    bundle_folders: tuple[tuple[str, Path], ...] = (  # (bundle name, folder): the shipped bundle first, then the chunk-size ablations
        ("shipped", BUILD_OUTPUT_FOLDER / "bundle"), ("target_128", BUILD_OUTPUT_FOLDER / "ablation_bundles/target_128"),
        ("target_256", BUILD_OUTPUT_FOLDER / "ablation_bundles/target_256"),
    )
    model_files_folder: Path = BUILD_OUTPUT_FOLDER / "model_files"  # the two pinned encoder models
    local_weights_path: Path = KAGGLE_FOLDER / "mixed_run_results/model.pth"  # the model live in the Space (checksum starts 528ef774)
    results_folder: Path = Path(__file__).resolve().parent / "results"  # generated result files


@dataclass(frozen=True)
class GoldenSetSettings:  # how the question sets are cut
    held_out_part_number: int = HELD_OUT_PART_NUMBER  # part holding the 816 answerable held-out questions
    block_count: int = 10  # equal contiguous blocks; even = development, odd = test
    block_edge_margin_characters: int = 2000  # questions this close to a block edge are dropped
    maximum_fact_overlap_f1: float = 0.6  # a fact-development question may not be closer than this to one of the 30
    evidence_window_characters: int = 2000  # accepted terms must appear this close to a fact-development question's evidence offset
    pair_sample_seed: int = 20261024  # seed of every sampling step of the golden sets and the calibration


@dataclass(frozen=True)
class RetrievalMeasureSettings:  # the retrieval ablation and its decision rules
    rank_cutoffs: tuple[int, ...] = (1, 3, 5, 10)  # hit-at-cutoff values reported
    ranking_depth: int = 10  # depth of mean reciprocal rank and normalized discounted cumulative gain
    packed_budgets: tuple[int, ...] = (300, 450, 600)  # context budgets for hit within the packed budget (chunk-size comparison)
    fusion_constants: tuple[int, ...] = (20, 100)  # extra fusion constants besides the shipped 60
    minimum_search_mode_gain: float = 0.02  # a single search must beat hybrid by this much hit-at-5 to replace it
    minimum_reranker_gain: float = 0.03  # the reranker needs this much hit-at-3 to be switched on
    minimum_chunk_size_gain: float = 0.02  # another chunk size must beat the shipped one by this much packed-budget hit to replace it
    chunk_comparison_budget: int = 600  # the budget the chunk-size decision reads
    space_slowdown_factor: float = 3.0  # Mac retrieval milliseconds times this estimate the Space (planning factor, DESIGN.md §11.6)
    reranker_off_limit_milliseconds: float = 400.0  # A9 ship gate without the reranker
    reranker_on_limit_milliseconds: float = 800.0  # A9 ship gate with the reranker


@dataclass(frozen=True)
class CalibrationSettings:  # evidence floor and pair threshold
    development_evidence_share: float = 0.90  # share of G1 development questions that must still get evidence
    fact_development_evidence_share: float = 0.70  # share of fact-development questions that must still get evidence (A7b)
    pair_sample_size: int = 3000  # training pairs searched leave-one-out
    pair_correct_f1: float = 0.5  # a neighbour is correct when its answer has at least this token F1 with the pair's own answer
    pair_precision_target: float = 0.95  # threshold choice on the first half
    minimum_selected_pairs: int = 50  # a threshold must select at least this many pairs, so one lucky pair cannot decide
    floor_rounding_digits: int = 3  # floors are rounded DOWN to this many decimals (more coverage, still readable)
    tradeoff_development_shares: tuple[float, ...] = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5)  # development coverage targets of the floor trade-off table


@dataclass(frozen=True)
class AnswerRunSettings:  # generation runs
    fact_sample_count: int = 3  # samples per fact question (2 right of 3 = right)
    held_out_sample_count: int = 1  # samples per G1 question
    seed_multiplier: int = 1000  # sample s of question q uses seed seed_multiplier * s + q
    max_new_tokens: int = 120  # as in the mixed run's fact evaluation
    temperature: float = 0.7  # sampling, as the Space and the Kaggle scoring use it
    top_p: float = 0.9  # sampling
    top_k: int = 40  # sampling
    repetition_penalty: float = 1.1  # sampling
    generator_time_limit_seconds: float = 600.0  # large, so no seeded answer is cut by the clock
    timing_probe_count: int = 30  # generations timed before the total is projected
    projection_limit_hours: float = 6.0  # above this the G1 development subset is halved
    generation_hard_cap: int = 3000  # never more generations than this in one run
    selection_layout_budget: int = 600  # context budget of the four-layout comparison
    reduced_budgets: tuple[int, ...] = (300, 450)  # budgets tried for the best layout


@dataclass(frozen=True)
class ShipGateSettings:  # the numbers of DESIGN.md §1.3 and §11.6
    facts_minimum: int = 15  # A1 of 30
    facts_maximum_turned_wrong: int = 2  # A1: of the questions the baseline got right
    facts_without_pairs_minimum: int = 12  # A2 of 27
    known_ten_minimum: int = 6  # A3 of 10
    passage_hit_at_5_minimum: float = 0.70  # A5 on G1 test
    unanswerable_evidence_maximum: float = 0.15  # A7: G3 questions that still get evidence
    development_evidence_minimum: float = 0.85  # A7: G1 development questions that get evidence
    fact_development_evidence_minimum: float = 0.70  # A7b
    pair_precision_minimum: float = 0.90  # A8 on the held-back half
    raft_facts_minimum: int = 20  # below this many of 30, a grounding fine-tune is needed
    raft_known_ten_minimum: int = 7  # below this many of 10
    raft_recall_gain_minimum: float = 0.10  # key-term recall gain over question_only on G1 test


@dataclass(frozen=True)
class EvaluationSettings:  # everything the command needs
    locations: DataLocations = field(default_factory=DataLocations)  # paths
    golden: GoldenSetSettings = field(default_factory=GoldenSetSettings)  # question sets
    retrieval: RetrievalMeasureSettings = field(default_factory=RetrievalMeasureSettings)  # ablation
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)  # floors and thresholds
    answers: AnswerRunSettings = field(default_factory=AnswerRunSettings)  # generation
    ship_gate: ShipGateSettings = field(default_factory=ShipGateSettings)  # gates
