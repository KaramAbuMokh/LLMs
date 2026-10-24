# The `calibrate` stage: evidence floor from the G1 development, fact-development and G3 evidence scores, pair threshold leave-one-out on 3,000
# training pairs. Writes results/calibration.json.
import dataclasses  # settings changes and result dictionaries
from typing import Any
from retrieval_contracts.bundle_manifest import PAIR_RECORDS_FILE_NAME  # pair file name
from retrieval_contracts.bundle_records import QuestionAnswerRecord, read_json_lines  # stored pairs
from retrieval_evaluation.evaluation_settings import EvaluationSettings  # numbers
from retrieval_evaluation.evaluation_stages.evaluation_runtime import EvaluationRuntime  # service objects
from retrieval_evaluation.evaluation_stages.retrieval_stage import build_chosen_search_settings  # chosen mode and reranker
from retrieval_evaluation.experiments.evaluation_report_writing import write_json_results  # result files
from retrieval_evaluation.experiments.threshold_calibration import (  # calibration steps
    build_floor_tradeoff, calibrate_evidence_floor, calibrate_pair_threshold, judge_pair_examples_on_facts, sample_training_pairs, score_leave_one_out,
)
from retrieval_evaluation.golden_sets.golden_question_records import GoldenSets  # question sets
from retrieval_evaluation.metrics.fact_rule_loading import FactRuleBook  # word rules

CALIBRATION_FILE_NAME = "calibration.json"  # inside the results folder
SHOW_EVERY_PAIR_THRESHOLD = -1.0  # makes every pair candidate visible during calibration


def build_calibrated_settings(chosen_settings: Any, calibration: dict[str, Any]) -> Any:  # chosen retrieval plus the calibrated floor and threshold
    floor_field = "rerank_evidence_floor" if chosen_settings.reranker_enabled else "passage_evidence_floor"  # the floor the gate reads
    threshold = calibration["pair_threshold"]["threshold"]  # None when the target was out of reach
    return dataclasses.replace(chosen_settings, **{floor_field: calibration["evidence_floor"]["floor"]},  # the calibrated floor
                               pair_examples_enabled=threshold is not None,  # pairs are on only when some threshold reached the precision target
                               pair_example_threshold=chosen_settings.pair_example_threshold if threshold is None else threshold)  # three changes


def run_calibration_stage(golden: GoldenSets, ablation: dict[str, Any], runtime: EvaluationRuntime, rule_book: FactRuleBook,  # floor and pair threshold from the chosen retrieval
                          settings: EvaluationSettings) -> dict[str, Any]:
    bundle_name = ablation["decisions"]["chunk_size_name"]  # the chunk size chosen by the retrieval stage
    chosen = build_chosen_search_settings(runtime.shipped_search_settings, ablation["decisions"])  # chosen mode and reranker
    retriever = runtime.build_retriever(bundle_name, dataclasses.replace(chosen, pair_examples_enabled=True, pair_example_threshold=SHOW_EVERY_PAIR_THRESHOLD))  # pair search on, all pairs visible
    best_scores = lambda questions: [retriever.retrieve(text).best_evidence_score for text in questions]  # gate score per question
    development_scores = best_scores(question.question for question in golden.held_out_development)  # G1 development
    fact_scores = best_scores(question.rule.question for question in golden.fact_development)  # fact development
    unanswerable_scores = best_scores(question.question for question in golden.no_answer)  # G3
    floor = calibrate_evidence_floor(development_scores, fact_scores, unanswerable_scores, settings.calibration)  # evidence floor
    pairs = read_json_lines(runtime.bundle_folders[bundle_name] / PAIR_RECORDS_FILE_NAME, QuestionAnswerRecord)  # the stored pairs
    sampled = sample_training_pairs(pairs, settings.calibration.pair_sample_size, settings.golden.pair_sample_seed)  # seeded 3,000
    neighbour_scores = score_leave_one_out(retriever, sampled, settings.calibration.pair_correct_f1)  # own pair removed
    pair_result = calibrate_pair_threshold(neighbour_scores, settings.calibration, settings.golden.pair_sample_seed)  # choose on one half, report on the other
    shown, correct = (0, 0) if pair_result.threshold is None else judge_pair_examples_on_facts(retriever, rule_book, [question.rule for question in golden.fact_development], pair_result.threshold)
    tradeoff = build_floor_tradeoff(development_scores, fact_scores, unanswerable_scores, settings.calibration.tradeoff_development_shares, settings.calibration.floor_rounding_digits)  # evidence floor against false alarms, per coverage target
    result = {"evidence_floor": dataclasses.asdict(floor), "floor_tradeoff": tradeoff, "pair_threshold": dataclasses.asdict(pair_result), "gate_score_kind": "rerank" if chosen.reranker_enabled else "dense",  # what the stage records
              "pair_fact_check": {"shown": shown, "correct": correct}, "sampled_pair_count": len(neighbour_scores)}
    write_json_results(settings.locations.results_folder / CALIBRATION_FILE_NAME, result)  # record
    return result  # for the next stage
