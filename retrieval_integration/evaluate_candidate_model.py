# Scores a candidate weights file (for example the grounding fine-tune) on the SAME final items and seeds as the evaluation's answers stage:
# fact-development, the 30 facts, the regression questions and the held-out test questions, with retrieval and with the plain prompt.
# Run from the project folder: PYTHONPATH=gpt2_service_huggingface_space:. python retrieval_integration/evaluate_candidate_model.py --weights W --layout L --budget B --name N
import argparse  # command line
import dataclasses  # settings copy and record conversion
import json  # result file
from pathlib import Path  # paths
from retrieval_evaluation.evaluation_settings import EvaluationSettings  # paths and numbers
from retrieval_evaluation.evaluation_stages.answers_stage import BASELINE_LAYOUT, build_lookup_helpers, score_final_arm, score_selection  # the stage's own scoring
from retrieval_evaluation.evaluation_stages.calibration_stage import CALIBRATION_FILE_NAME, build_calibrated_settings  # calibrated retrieval
from retrieval_evaluation.evaluation_stages.golden_sets_stage import load_golden_sets, load_rule_book  # question sets and word rules
from retrieval_evaluation.evaluation_stages.retrieval_stage import ABLATION_FILE_NAME, build_chosen_search_settings  # chosen retrieval
from retrieval_evaluation.experiments.answer_run_planning import AnswerConfiguration, GenerationClock, build_answer_items, build_configuration_name  # plan records
from retrieval_evaluation.experiments.answer_runs import AnswerRunTools, run_answer_items  # the runner
from retrieval_evaluation.experiments.evaluation_report_writing import read_json_results  # result files
from retrieval_evaluation.run_evaluation_command import build_evaluation_runtime  # wires the service objects (heavy imports)

RESULTS_FOLDER = Path(__file__).resolve().parent / "results"  # private result files


def evaluate_candidate(weights_path: Path, layout_name: str, budget: int, name: str, decisions_folder: Path | None = None) -> dict:  # runs both arms and returns the scores
    settings = EvaluationSettings()  # shipped evaluation numbers
    settings = dataclasses.replace(settings, locations=dataclasses.replace(settings.locations, local_weights_path=weights_path))  # the candidate's weights
    golden, rule_book = load_golden_sets(settings), load_rule_book(settings)  # sets and judge
    results = decisions_folder or settings.locations.results_folder  # where the retrieval and calibration decisions are (the main results unless told otherwise)
    ablation, calibration = read_json_results(results / ABLATION_FILE_NAME), read_json_results(results / CALIBRATION_FILE_NAME)  # earlier decisions
    runtime = build_evaluation_runtime(settings, True)  # service objects with the candidate model
    bundle_name = ablation["decisions"]["chunk_size_name"]  # the chosen passage size
    calibrated = build_calibrated_settings(build_chosen_search_settings(runtime.shipped_search_settings, ablation["decisions"]), calibration)  # as shipped
    find_texts, find_problems = build_lookup_helpers(runtime.bundle_folders[bundle_name])  # text lookup and the copyright check
    tools = AnswerRunTools(runtime.build_retriever(bundle_name, calibrated), runtime.answer_service, runtime.build_answer_request, runtime.make_random_generator, find_texts, find_problems)  # one tool bundle for the whole run
    arms = [AnswerConfiguration(build_configuration_name(layout_name, budget), runtime.layouts[layout_name], budget),  # with retrieval
            AnswerConfiguration(build_configuration_name(BASELINE_LAYOUT, budget), runtime.layouts[BASELINE_LAYOUT], budget)]  # the candidate's plain path
    samples = settings.answers.fact_sample_count  # samples per fact question
    fact_development = build_answer_items("fact_development", [(q.question_id, q.rule.question) for q in golden.fact_development], samples)  # FD
    final_items = (build_answer_items("fact", [(q.question_id, q.rule.question) for q in golden.fact], samples)  # G2
                   + build_answer_items("regression", [(q.question_id, q.rule.question) for q in golden.regression], samples)  # G4
                   + build_answer_items("held_out_test", [(q.question_id, q.question) for q in golden.held_out_test], settings.answers.held_out_sample_count))  # G1 test
    records = run_answer_items(fact_development + final_items, arms, tools, GenerationClock(), settings.answers.seed_multiplier, runtime.report_progress)  # every generation
    final_scores = {arm.name: score_final_arm([r for r in records if r.configuration_name == arm.name and r.set_name != "fact_development"], golden, rule_book) for arm in arms}  # per arm
    development_scores = {arm.name: dataclasses.asdict(score_selection([r for r in records if r.set_name == "fact_development"], arm, golden, rule_book)) for arm in arms}  # FD majority
    return {"name": name, "weights_path": str(weights_path), "final": final_scores, "fact_development": development_scores, "records": [dataclasses.asdict(r) for r in records]}  # scores and every generated answer


if __name__ == "__main__":  # run as a script
    parser = argparse.ArgumentParser(description="Score a candidate weights file on the evaluation's final items")  # parser
    parser.add_argument("--weights", required=True, type=Path, help="candidate model.pth")  # weights
    parser.add_argument("--layout", default="passages_before_template", help="prompt layout with retrieval")  # layout
    parser.add_argument("--budget", type=int, required=True, help="context token budget with retrieval")  # budget
    parser.add_argument("--name", required=True, help="result file name stem")  # output name
    parser.add_argument("--decisions-folder", type=Path, default=None, help="folder with retrieval_ablation.json and calibration.json (default: the main results)")  # decisions
    options = parser.parse_args()  # parsed
    record = evaluate_candidate(options.weights, options.layout, options.budget, options.name, options.decisions_folder)  # run
    (RESULTS_FOLDER / ("candidate_%s.json" % options.name)).write_text(json.dumps(record, default=str), encoding="utf-8")  # private record
    print(json.dumps({"final": {arm: {key: scores[key] for key in ("known_ten", "facts_without_pairs", "iron_throne_names_mountain")} | {"facts_right": scores["facts"]["majority_right"]} for arm, scores in record["final"].items()}}, indent=1))  # short summary on screen
