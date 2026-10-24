# The retrieval and calibration stages end to end on the fixture: fake retrievers, real stage code, real result files
from retrieval_evaluation.evaluation_stages.calibration_stage import build_calibrated_settings, run_calibration_stage  # stage 3
from retrieval_evaluation.evaluation_stages.retrieval_stage import ABLATION_FILE_NAME, build_chosen_search_settings, run_retrieval_stage  # stage 2
from retrieval_evaluation.experiments.evaluation_report_writing import read_json_results  # result files
from retrieval_evaluation.golden_sets.golden_question_records import golden_sets_from_dict, golden_sets_to_dict  # JSON round trip of the sets
from retrieval_evaluation.metrics.fact_rule_loading import load_fact_rule_book  # the 30 rules
from retrieval_evaluation.tests.support.fixture_retrieval_fakes import FakeSearchSettings  # fake settings
from retrieval_evaluation.tests.support.fixture_stage_inputs import build_fake_runtime, build_fixture_golden_sets, build_fixture_settings  # fixture inputs


def test_retrieval_stage_decides_hybrid_without_reranker_and_writes_its_result(tmp_path):  # fixture rankings are equal across modes
    settings = build_fixture_settings(tmp_path)  # results go to a temporary folder
    result = run_retrieval_stage(build_fixture_golden_sets(), build_fake_runtime(), settings)  # all four fixture questions on both splits
    assert result["decisions"]["passage_search_mode"] == "hybrid" and result["decisions"]["reranker_enabled"] is False  # nothing beats the default
    assert result["decisions"]["chunk_size_name"] == "shipped" and result["decisions"]["fusion_constant"] == 60  # only one bundle, constant kept
    assert [variant["name"] for variant in result["variants"]][0] == "hybrid" and len(result["variants"]) == 6  # the six shipped-bundle variants
    assert result["test_run"]["question_count"] == 4 and result["test_run"]["metrics"]["hit_at_5"] == 1.0  # the chosen retrieval, run once on test
    assert read_json_results(tmp_path / ABLATION_FILE_NAME) == result or read_json_results(tmp_path / ABLATION_FILE_NAME)["decisions"] == result["decisions"]  # written


def test_calibration_stage_separates_the_groups_and_reports_no_pair_threshold(tmp_path):  # fixture pairs are unrelated to each other
    settings, golden, runtime = build_fixture_settings(tmp_path), build_fixture_golden_sets(), build_fake_runtime()  # inputs
    ablation = run_retrieval_stage(golden, runtime, settings)  # the stage before
    locations = settings.locations  # source paths for the real rule book
    result = run_calibration_stage(golden, ablation, runtime, load_fact_rule_book(locations.fact_rules_path, locations.run_settings_path), settings)
    floor = result["evidence_floor"]  # the evidence floor result
    assert 0.0976 < floor["floor"] <= 0.178 and floor["unanswerable_share"] == 0.0 and floor["development_share"] == 1.0  # fg-5 and fg-6 stay out
    assert result["pair_threshold"]["threshold"] is None and result["sampled_pair_count"] == 4  # four pairs, none ever a good neighbour
    calibrated = build_calibrated_settings(build_chosen_search_settings(FakeSearchSettings(), ablation["decisions"]), result)  # service settings for the answers stage
    assert calibrated.passage_evidence_floor == floor["floor"] and calibrated.pair_examples_enabled is False  # applied, pairs off


def test_golden_sets_json_round_trip_keeps_the_fixture_sets_intact():  # what the golden stage writes
    golden = build_fixture_golden_sets()  # built
    assert golden_sets_from_dict(golden_sets_to_dict(golden)) == golden  # identical after the round trip
