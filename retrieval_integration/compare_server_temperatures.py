# Scores the live (grounded) model WITH retrieval at one server temperature on chosen fact sets, with the evaluation's own seeds and word rules,
# to decide the default temperature of the Space. Tune on fact-development ("fact_development"), confirm once on the 30 facts ("fact") and the old mistakes ("regression").
# Run from the project folder: PYTHONPATH=<service folder>:. python retrieval_integration/compare_server_temperatures.py --weights W --temperature 0.5 --sets fact_development --name t05
import argparse  # command line
import dataclasses  # settings copy and record conversion
import json  # result file
from pathlib import Path  # paths
from retrieval_evaluation.evaluation_settings import EvaluationSettings  # paths and numbers
from retrieval_evaluation.evaluation_stages.answers_stage import build_lookup_helpers  # text lookup and the copyright check
from retrieval_evaluation.evaluation_stages.calibration_stage import CALIBRATION_FILE_NAME, build_calibrated_settings  # calibrated retrieval
from retrieval_evaluation.evaluation_stages.golden_sets_stage import load_golden_sets, load_rule_book  # question sets and word rules
from retrieval_evaluation.evaluation_stages.retrieval_stage import ABLATION_FILE_NAME, build_chosen_search_settings  # chosen retrieval
from retrieval_evaluation.experiments.answer_run_planning import AnswerConfiguration, GenerationClock, build_answer_items, build_configuration_name  # plan records
from retrieval_evaluation.experiments.answer_runs import AnswerRunTools, run_answer_items  # the runner
from retrieval_evaluation.experiments.evaluation_report_writing import read_json_results  # result files
from retrieval_evaluation.metrics.answer_run_scoring import score_fact_records  # the majority rule
from retrieval_evaluation.run_evaluation_command import build_evaluation_runtime  # wires the service objects (heavy imports)

RESULTS_FOLDER = Path(__file__).resolve().parent / "results"  # private result files
SHIPPED_LAYOUT_NAME = "passages_before_template"  # the layout the Space uses
SHIPPED_BUDGET = 600  # retrieved tokens the Space allows
FACT_SET_NAMES = ("fact_development", "fact", "regression")  # the three fact-style question sets


def list_set_questions(golden, set_name: str) -> list[tuple[str, str]]:  # (question id, question text) pairs of one fact-style set
    return [(question.question_id, question.rule.question) for question in getattr(golden, set_name)]  # every set stores its word rule with the question


def score_set(records, golden, rule_book, set_name: str) -> dict:  # majority score of one set, as a plain dictionary
    rules = {question.question_id: question.rule for question in getattr(golden, set_name)}  # word rules by question id
    set_records = [record for record in records if record.set_name == set_name]  # this set's answers only
    score = score_fact_records(set_records, rule_book, rules)  # 2 of 3 samples right = right
    guard_cut_count = sum(record.verbatim_guard_cut for record in set_records)  # answers the copy guard shortened
    return {**dataclasses.asdict(score), "guard_cut_count": guard_cut_count, "mean_answer_tokens": sum(r.answer_token_count for r in set_records) / max(1, len(set_records))}  # score plus two health numbers


def evaluate_temperature(weights_path: Path, temperature: float, set_names: tuple[str, ...]) -> dict:  # generates every sample of the chosen sets and scores them
    settings = EvaluationSettings()  # shipped evaluation numbers
    settings = dataclasses.replace(  # the candidate's weights and the temperature under test
        settings, locations=dataclasses.replace(settings.locations, local_weights_path=weights_path),  # the candidate weights file
        answers=dataclasses.replace(settings.answers, temperature=temperature),  # the temperature under test
    )
    golden, rule_book = load_golden_sets(settings), load_rule_book(settings)  # question sets and word rules
    ablation, calibration = (read_json_results(RESULTS_FOLDER / "reranker_variant" / name) for name in (ABLATION_FILE_NAME, CALIBRATION_FILE_NAME))  # the reranker decisions that are live
    runtime = build_evaluation_runtime(settings, True)  # service objects with the candidate model
    bundle_name = ablation["decisions"]["chunk_size_name"]  # the chosen passage size
    calibrated = build_calibrated_settings(build_chosen_search_settings(runtime.shipped_search_settings, ablation["decisions"]), calibration)  # retrieval as shipped
    find_texts, find_problems = build_lookup_helpers(runtime.bundle_folders[bundle_name])  # text lookup and the copyright check
    tools = AnswerRunTools(runtime.build_retriever(bundle_name, calibrated), runtime.answer_service, runtime.build_answer_request, runtime.make_random_generator, find_texts, find_problems)  # one tool bundle
    arm = AnswerConfiguration(build_configuration_name(SHIPPED_LAYOUT_NAME, SHIPPED_BUDGET), runtime.layouts[SHIPPED_LAYOUT_NAME], SHIPPED_BUDGET)  # the shipped prompt
    items = [item for set_name in set_names for item in build_answer_items(set_name, list_set_questions(golden, set_name), settings.answers.fact_sample_count)]  # every sample to generate
    records = run_answer_items(items, [arm], tools, GenerationClock(), settings.answers.seed_multiplier, runtime.report_progress)  # all generations
    return {"temperature": temperature, "weights_path": str(weights_path), "scores": {name: score_set(records, golden, rule_book, name) for name in set_names},  # per-set scores
            "records": [dataclasses.asdict(record) for record in records]}  # every generated answer


if __name__ == "__main__":  # run as a script
    parser = argparse.ArgumentParser(description="Score the grounded model with retrieval at one temperature")  # parser
    parser.add_argument("--weights", required=True, type=Path, help="candidate model.pth (the live grounding fine-tune)")  # weights
    parser.add_argument("--temperature", required=True, type=float, help="server default temperature under test")  # temperature
    parser.add_argument("--sets", nargs="+", choices=FACT_SET_NAMES, default=["fact_development"], help="question sets to run")  # sets
    parser.add_argument("--threads", type=int, default=3, help="torch threads, so several runs can share the Mac")  # threads
    parser.add_argument("--name", required=True, help="result file name stem")  # output name
    options = parser.parse_args()  # parsed
    import torch  # thread setting
    torch.set_num_threads(options.threads)  # keep parallel runs from fighting over cores
    record = evaluate_temperature(options.weights, options.temperature, tuple(options.sets))  # run
    (RESULTS_FOLDER / ("temperature_%s.json" % options.name)).write_text(json.dumps(record, default=str), encoding="utf-8")  # private record
    print(json.dumps({"temperature": record["temperature"], "scores": record["scores"]}, indent=1))  # shown to the caller
