# The report: tables, the G1 inflation sentence, A5's margin, skipped stages, and the result file helpers
import pytest  # approximate comparisons
from retrieval_evaluation.experiments.evaluation_report_writing import (  # under test
    INFLATION_NOTICE, compose_evaluation_report, compute_margin_points, format_markdown_table, read_json_results, write_json_results,
)


def test_markdown_table_has_header_rule_and_rows():  # shape
    assert format_markdown_table(("first", "second"), [(1, 2)]) == ["| first | second |", "| --- | --- |", "| 1 | 2 |", ""]  # exact lines


def test_margin_of_a_rate_over_196_questions_is_about_six_points():  # A5's margin, DESIGN.md §2
    assert compute_margin_points(0.75, 196) == pytest.approx(6.07, abs=0.05)  # 1.96 x sqrt(0.75 x 0.25 / 196) x 100
    assert compute_margin_points(0.5, 0) == 0.0  # no questions


def test_report_skips_stages_that_did_not_run_and_states_the_inflation():  # partial report
    report = compose_evaluation_report({"golden": {"counts": {"held_out_development": 189, "held_out_test": 196}}, "ablation": None})  # only golden ran
    assert "## Question sets" in report and "189" in report and INFLATION_NOTICE in report  # present
    assert "Retrieval ablation" not in report and "Ship gate" not in report  # skipped


def test_full_report_has_every_section(tmp_path):  # all five sections from minimal stage dictionaries
    metrics = {key: 0.5 for key in ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mean_reciprocal_rank_at_10", "normalized_discounted_cumulative_gain_at_10",
                                    "hit_within_packed_budget_300", "hit_within_packed_budget_450", "hit_within_packed_budget_600")}  # all keys read
    variant = {"name": "hybrid", "bundle_name": "shipped", "metrics": metrics, "p95_total_milliseconds": 12.0, "question_count": 196}  # one variant
    arm = {"facts": {"majority_right": 20}, "known_ten": 8, "facts_without_pairs": 17, "copyright_problem_count": 0,
           "held_out": {"key_term_recall": 0.3, "unsupported_name_rate": 0.1, "no_evidence_rate": 0.2}}  # one final arm
    results = {"golden": {"counts": {"first_count": 1}},
               "ablation": {"variants": [variant], "chunk_sizes": [variant], "test_run": variant, "decisions": {"passage_search_mode": "hybrid", "reranker_enabled": False,
                            "reranker_reason": "r", "chunk_size_name": "shipped"}},
               "calibration": {"evidence_floor": {"floor": 0.6, "development_share": 0.9, "fact_development_share": 0.7, "unanswerable_share": 0.1}, "gate_score_kind": "dense",
                               "pair_threshold": {"threshold": 0.85, "held_back_precision": 0.95, "held_back_coverage": 0.2}, "pair_fact_check": {"shown": 3, "correct": 3}},
               "answers": {"selection": [{"name": "c", "fact_development_majority_right": 20, "held_out_key_term_recall": 0.3, "mean_prompt_tokens": 500}], "chosen": "c",
                           "plan": {"reason": "full held-out set fits the caps"}, "final": {"baseline": arm, "chosen": arm}},
               "recommended": {"ship_gate": [{"gate": "A1", "status": "pass", "detail": "ok"}], "raft_needed": True, "raft_reasons": ["x"]}}
    report = compose_evaluation_report(results)  # all sections
    for heading in ("## Question sets", "## Retrieval ablation", "## Calibration", "## Answer runs", "## Ship gate"):  # every section
        assert heading in report  # present
    assert "margin about +/- 7.0 points" in report and "Grounding fine-tune: needed (x)" in report  # A5 margin and the RAFT line


def test_result_files_round_trip(tmp_path):  # write then read
    path = tmp_path / "nested" / "result.json"  # the folder does not exist yet
    assert read_json_results(path) is None  # not run
    write_json_results(path, {"second": 1, "first": [1, 2]})  # write
    assert read_json_results(path) == {"first": [1, 2], "second": 1}  # read back
