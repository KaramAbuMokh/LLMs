# The decision rules of DESIGN.md §11.6 on hand-made numbers
from retrieval_evaluation.evaluation_settings import RetrievalMeasureSettings, ShipGateSettings  # rule numbers
from retrieval_evaluation.experiments.settings_recommendation import (  # under test
    ConfigurationSelectionScore, ShipMeasurements, choose_chunk_size_name, choose_layout_and_budget, choose_search_mode, decide_raft, decide_reranker, evaluate_ship_gate,
)

MEASURE, GATE = RetrievalMeasureSettings(), ShipGateSettings()  # shipped numbers


def test_hybrid_stays_unless_a_single_search_wins_by_two_points():  # D7
    assert choose_search_mode({"hybrid": 0.80, "dense_only": 0.81, "lexical_only": 0.79}, MEASURE) == "hybrid"  # one point is not enough
    assert choose_search_mode({"hybrid": 0.80, "dense_only": 0.82, "lexical_only": 0.79}, MEASURE) == "dense_only"  # exactly two points
    assert choose_search_mode({"hybrid": 0.70, "dense_only": 0.72, "lexical_only": 0.80}, MEASURE) == "lexical_only"  # the better single search


def test_reranker_needs_three_points_and_a_fast_enough_estimate():  # D8 and A9
    assert decide_reranker(0.70, 0.73, 200.0, MEASURE)[0] is True  # +3 points, 600 ms estimated on the Space
    assert decide_reranker(0.70, 0.72, 100.0, MEASURE)[0] is False  # only +2 points
    assert decide_reranker(0.70, 0.80, 300.0, MEASURE)[0] is False  # 900 ms estimated: A9 fails


def test_chunk_size_changes_only_for_a_clear_gain():  # shipped size is kept otherwise
    assert choose_chunk_size_name({"shipped": 0.80, "target_128": 0.81, "target_256": 0.79}, "shipped", MEASURE) == "shipped"  # one point
    assert choose_chunk_size_name({"shipped": 0.80, "target_128": 0.83, "target_256": 0.79}, "shipped", MEASURE) == "target_128"  # three points


def test_layout_choice_uses_majority_then_recall_then_shorter_prompts():  # rule 3
    score = lambda name, right, recall, tokens: ConfigurationSelectionScore(name, name, 600, right, recall, tokens)  # shorthand
    assert choose_layout_and_budget([score("a", 20, 0.3, 500), score("b", 22, 0.1, 700)]).name == "b"  # majority first
    assert choose_layout_and_budget([score("a", 22, 0.3, 700), score("b", 22, 0.2, 400)]).name == "a"  # then recall
    assert choose_layout_and_budget([score("a", 22, 0.3, 700), score("b", 22, 0.3, 400)]).name == "b"  # then fewer tokens


def test_raft_triggers():  # any of three
    assert decide_raft(21, 8, 0.15, GATE) == (False, [])  # all fine
    needed, reasons = decide_raft(19, 6, 0.05, GATE)  # all three fire
    assert needed and len(reasons) == 3  # three reasons


def test_ship_gate_passes_good_numbers_and_flags_each_failure():  # A1-A5, A7, A7b, A8, A16, recall
    good = ShipMeasurements(21, 17, 8, 1, False, 0.78, 0.10, 0.92, 0.75, 0.93, 0, 0.30, 0.10)  # every gate satisfied
    results = {result.gate: result.status for result in evaluate_ship_gate(good, GATE)}  # by name
    assert all(results[name] == "pass" for name in ("A1", "A2", "A3", "A4", "A5", "A7", "A7b", "A8", "A16", "key_term_recall"))  # all pass
    assert [results[name] for name in ("A13", "A14", "A15")] == ["live_check"] * 3  # confirmed on the Space
    bad = ShipMeasurements(14, 11, 5, 3, True, 0.65, 0.20, 0.80, 0.60, None, 2, 0.05, 0.10)  # every gate broken
    failed = {result.gate for result in evaluate_ship_gate(bad, GATE) if result.status == "fail"}  # failures
    assert failed == {"A1", "A2", "A3", "A4", "A5", "A7", "A7b", "A8", "A16", "key_term_recall"}  # all ten
