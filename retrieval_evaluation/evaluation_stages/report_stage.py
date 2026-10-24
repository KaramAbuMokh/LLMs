# The report stage: reads the result files of the earlier stages, applies the ship gate and the grounding fine-tune rule, writes
# results/recommended_settings.json and results/EVALUATION-REPORT.md. Stages that have not run are left out of the report.
import dataclasses  # gate results to dictionaries
from typing import Any, Mapping
from retrieval_evaluation.evaluation_settings import EvaluationSettings  # gate numbers and paths
from retrieval_evaluation.evaluation_stages.answers_stage import ANSWERS_FILE_NAME  # result file names of the earlier stages
from retrieval_evaluation.evaluation_stages.calibration_stage import CALIBRATION_FILE_NAME
from retrieval_evaluation.evaluation_stages.golden_sets_stage import GOLDEN_FILE_NAME  # golden stage file name
from retrieval_evaluation.evaluation_stages.retrieval_stage import ABLATION_FILE_NAME  # retrieval stage file name
from retrieval_evaluation.experiments.evaluation_report_writing import compose_evaluation_report, read_json_results, write_json_results  # writing
from retrieval_evaluation.experiments.settings_recommendation import ShipMeasurements, build_recommended_settings, decide_raft, evaluate_ship_gate  # rules

RECOMMENDED_FILE_NAME = "recommended_settings.json"  # inside the results folder
REPORT_FILE_NAME = "EVALUATION-REPORT.md"  # inside the results folder


def build_ship_measurements(ablation: Mapping[str, Any], calibration: Mapping[str, Any], answers: Mapping[str, Any]) -> ShipMeasurements:  # numbers for the ship gate
    chosen, baseline = answers["final"]["chosen"], answers["final"]["baseline"]  # the two arms
    turned_wrong = len(set(baseline["facts"]["right_question_ids"]) - set(chosen["facts"]["right_question_ids"]))  # right before, wrong now
    floor, pair = calibration["evidence_floor"], calibration["pair_threshold"]  # calibration results
    return ShipMeasurements(
        chosen["facts"]["majority_right"], chosen["facts_without_pairs"], chosen["known_ten"], turned_wrong, chosen["iron_throne_names_mountain"],
        ablation["test_run"]["metrics"]["hit_at_5"], floor["unanswerable_share"], floor["development_share"], floor["fact_development_share"],
        pair["held_back_precision"], sum(len(record["copyright_problems"]) for record in answers["records"]),  # A16 over EVERY answer of the run
        chosen["held_out"]["key_term_recall"], baseline["held_out"]["key_term_recall"],
    )


def build_recommended(ablation: Mapping[str, Any], calibration: Mapping[str, Any], answers: Mapping[str, Any], settings: EvaluationSettings) -> dict[str, Any]:  # the settings the integrator copies
    measured = build_ship_measurements(ablation, calibration, answers)  # numbers for the gate
    raft_needed, raft_reasons = decide_raft(measured.facts_right, measured.known_ten_right, measured.chosen_key_term_recall - measured.baseline_key_term_recall,
                                            settings.ship_gate)  # rule 5
    chosen_score = next(score for score in answers["selection"] if score["name"] == answers["chosen"])  # layout and budget of the winner
    uses_rerank_floor = calibration["gate_score_kind"] == "rerank"  # which floor the service gate reads
    floor_value = calibration["evidence_floor"]["floor"]  # calibrated floor
    decisions = {**{key: ablation["decisions"][key] for key in ("passage_search_mode", "reranker_enabled", "fusion_constant", "chunk_size_name")},
                 "passage_evidence_floor": None if uses_rerank_floor else floor_value, "rerank_evidence_floor": floor_value if uses_rerank_floor else None,
                 "pair_examples_enabled": calibration["pair_threshold"]["threshold"] is not None, "pair_example_threshold": calibration["pair_threshold"]["threshold"],  # pairs on only with a calibrated threshold
                 "layout": chosen_score["layout_name"],
                 "context_token_budget": chosen_score["context_token_budget"], "raft_needed": raft_needed, "raft_reasons": raft_reasons,
                 "ship_gate": [dataclasses.asdict(gate) for gate in evaluate_ship_gate(measured, settings.ship_gate)]}  # None = keep the shipped default
    return build_recommended_settings(decisions, {"measurements": dataclasses.asdict(measured), "reranker_reason": ablation["decisions"]["reranker_reason"]})


def run_report_stage(settings: EvaluationSettings) -> str:  # writes the recommended settings (when every stage ran) and the report
    folder = settings.locations.results_folder  # where the stages left their files
    golden, ablation, calibration, answers = (read_json_results(folder / name) for name in (GOLDEN_FILE_NAME, ABLATION_FILE_NAME, CALIBRATION_FILE_NAME, ANSWERS_FILE_NAME))
    recommended = build_recommended(ablation, calibration, answers, settings) if ablation and calibration and answers else None  # needs the three stages
    if recommended:  # only a complete evaluation recommends settings
        write_json_results(folder / RECOMMENDED_FILE_NAME, recommended)  # for the integrator
    report = compose_evaluation_report({"golden": golden, "ablation": ablation, "calibration": calibration, "answers": answers, "recommended": recommended})
    (folder / REPORT_FILE_NAME).write_text(report, encoding="utf-8")  # the human-readable report
    return report  # for printing
