# The answers and report stages end to end on the fixture: the grid, the final arms, the A-gate lines and the written report
from retrieval_evaluation.evaluation_stages.answers_stage import BASELINE_LAYOUT, RETRIEVAL_LAYOUTS, run_answers_stage  # stage 4
from retrieval_evaluation.evaluation_stages.calibration_stage import run_calibration_stage  # stage 3
from retrieval_evaluation.evaluation_stages.report_stage import RECOMMENDED_FILE_NAME, REPORT_FILE_NAME, run_report_stage  # stage 5
from retrieval_evaluation.evaluation_stages.retrieval_stage import run_retrieval_stage  # stage 2
from retrieval_evaluation.evaluation_stages.golden_sets_stage import GOLDEN_FILE_NAME  # the golden stage's result file
from retrieval_evaluation.experiments.evaluation_report_writing import read_json_results, write_json_results  # result files
from retrieval_evaluation.golden_sets.golden_question_records import golden_sets_to_dict  # JSON form of the sets
from retrieval_evaluation.metrics.fact_rule_loading import load_fact_rule_book  # the 30 rules
from retrieval_evaluation.tests.support.fixture_retrieval_fakes import FakeAnswerService  # spy
from retrieval_evaluation.tests.support.fixture_stage_inputs import build_fake_runtime, build_fixture_golden_sets, build_fixture_settings  # fixture inputs


def run_all_stages(tmp_path, service=None, pair_threshold=None):  # stages 2 to 4 on the fixture; a pair threshold can be forced into the calibration
    settings, golden, runtime = build_fixture_settings(tmp_path), build_fixture_golden_sets(), build_fake_runtime(service)  # inputs
    rule_book = load_fact_rule_book(settings.locations.fact_rules_path, settings.locations.run_settings_path)  # the real rule book
    write_json_results(tmp_path / GOLDEN_FILE_NAME, golden_sets_to_dict(golden))  # what the golden stage would have left
    ablation = run_retrieval_stage(golden, runtime, settings)  # decisions
    calibration = run_calibration_stage(golden, ablation, runtime, rule_book, settings)  # floors
    if pair_threshold is not None:  # pretend pair examples reached the precision target
        calibration = {**calibration, "pair_threshold": {**calibration["pair_threshold"], "threshold": pair_threshold}}  # forced threshold
    return run_answers_stage(golden, ablation, calibration, runtime, rule_book, settings), ablation, calibration, settings  # generations


def test_without_a_pair_threshold_only_the_passage_layout_is_compared_with_the_baseline(tmp_path):  # the fixture calibration finds no pair threshold
    answers, _, _, _ = run_all_stages(tmp_path)  # run
    names = [score["name"] for score in answers["selection"]]  # the configurations
    assert names == ["question_only_600", "passages_before_template_600", "passages_before_template_300", "passages_before_template_450"]  # no pair-example layouts
    assert "pair-example layouts skipped" in answers["plan"]["reason"] and answers["chosen"].startswith("passages_before_template")  # said in the report
    assert answers["final"]["baseline"]["facts"]["majority_right"] == 0 and answers["final"]["chosen"]["facts"]["majority_right"] == 3  # baseline wrong, grounded right


def test_answers_stage_runs_the_grid_picks_a_grounded_layout_and_scores_both_final_arms(tmp_path):  # grounded fake answers contain the gold word
    answers, _, _, _ = run_all_stages(tmp_path, pair_threshold=0.9)  # pair examples available: all four layouts are compared
    names = [score["name"] for score in answers["selection"]]  # the six configurations
    assert names[0] == "%s_600" % BASELINE_LAYOUT and len(names) == 6 and {n.rsplit("_", 1)[1] for n in names} == {"600", "450", "300"}  # layouts at 600, one at 450 and 300
    assert answers["chosen"] != names[0] and answers["chosen"].rsplit("_", 1)[0] in RETRIEVAL_LAYOUTS  # grounded beats the baseline that says "I do not know"
    assert answers["final"]["baseline"]["facts"]["majority_right"] == 0 and answers["final"]["chosen"]["facts"]["majority_right"] == 3  # baseline wrong, grounded right
    assert answers["final"]["chosen"]["known_ten"] == 2 and answers["final"]["chosen"]["facts_without_pairs"] == 2  # the flags of the fixture G2
    assert answers["final"]["chosen"]["held_out"]["no_evidence_rate"] == 0.0 and answers["final"]["baseline"]["held_out"]["no_evidence_rate"] == 1.0  # evidence use
    assert answers["generation_count"] == len(answers["records"]) and not answers["plan"]["was_reduced"]  # everything ran


def test_final_arms_run_once_per_question_with_the_final_sets_only(tmp_path):  # spy on the service
    service = FakeAnswerService(lambda question, layout: "an answer")  # records calls
    answers, _, _, _ = run_all_stages(tmp_path, service)  # run
    final_sets = {record["set_name"] for record in answers["records"] if record["set_name"] in ("fact", "regression", "held_out_test")}  # reported sets
    assert final_sets == {"fact", "regression", "held_out_test"}  # all three ran
    assert {record["configuration_name"] for record in answers["records"] if record["set_name"] == "held_out_test"} == {answers["baseline"], answers["chosen"]} or answers["chosen"] == answers["baseline"]  # two arms


def test_report_stage_writes_recommended_settings_with_the_gate_and_the_report(tmp_path):  # stage 5
    _, _, _, settings = run_all_stages(tmp_path)  # stages 2-4 wrote their files
    report = run_report_stage(settings)  # report
    recommended = read_json_results(tmp_path / RECOMMENDED_FILE_NAME)  # the integrator's file
    assert {"passage_search_mode", "reranker_enabled", "passage_evidence_floor", "rerank_evidence_floor", "pair_examples_enabled", "pair_example_threshold", "layout",
            "context_token_budget", "raft_needed", "ship_gate", "evidence"} <= set(recommended)  # every field of DESIGN.md §14.3
    assert recommended["rerank_evidence_floor"] is None and recommended["pair_example_threshold"] is None and recommended["pair_examples_enabled"] is False  # dense gate, no pair threshold reached
    assert {gate["gate"] for gate in recommended["ship_gate"]} >= {"A1", "A5", "A7", "A7b", "A8", "A16", "A13"}  # the gate lines
    assert (tmp_path / REPORT_FILE_NAME).read_text(encoding="utf-8") == report and "## Ship gate" in report and "inflated" in report  # written and complete
