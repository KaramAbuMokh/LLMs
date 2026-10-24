# Calibration: the fixture floor separates fg-1..4 from fg-5/6; leave-one-out never returns a pair as its own neighbour; threshold choice and halves
import dataclasses  # settings copies
import pytest  # raises and approximate comparisons
from retrieval_contracts.bundle_manifest import PAIR_RECORDS_FILE_NAME  # fixture pair file
from retrieval_contracts.bundle_records import QuestionAnswerRecord, read_json_lines  # stored pairs
from retrieval_contracts.shared_fixture.fixture_files import FIXTURE_BUNDLE_FOLDER, load_fixture_golden_questions  # fixture
from retrieval_evaluation.evaluation_settings import CalibrationSettings  # numbers
from retrieval_evaluation.experiments.threshold_calibration import (  # under test
    NeighbourScore, calibrate_evidence_floor, calibrate_pair_threshold, choose_pair_threshold, find_leave_one_out_neighbour,
    judge_pair_examples_on_facts, measure_precision, pick_highest_floor, round_down, sample_training_pairs, score_leave_one_out,
)
from retrieval_evaluation.metrics.fact_rule_loading import FactRule, FactRuleBook  # word rules
from retrieval_evaluation.tests.support.fixture_retrieval_fakes import FakeScoredPair, FakeSearchSettings, FixtureRetriever  # fakes

PAIRS = read_json_lines(FIXTURE_BUNDLE_FOLDER / PAIR_RECORDS_FILE_NAME, QuestionAnswerRecord)  # the four fixture pairs


def test_fixture_floor_separates_the_answerable_questions_from_fg5_and_fg6():  # DESIGN.md §14.3
    retriever = FixtureRetriever(FakeSearchSettings())  # recorded rankings
    scores = {golden_question.question_id: retriever.retrieve(golden_question.question).best_evidence_score for golden_question in load_fixture_golden_questions()}  # best dense score per question
    answerable = [scores[name] for name in ("fg-1", "fg-2", "fg-3", "fg-4")]  # must keep evidence
    result = calibrate_evidence_floor(answerable, answerable, [scores["fg-5"], scores["fg-6"]], CalibrationSettings())  # calibrated
    assert max(scores["fg-5"], scores["fg-6"]) < result.floor <= min(answerable)  # strictly between the two groups
    assert (result.development_share, result.fact_development_share, result.unanswerable_share) == (1.0, 1.0, 0.0)  # all in, all out


def test_pick_highest_floor_handles_shares_and_missing_scores():  # order statistics
    assert pick_highest_floor([0.5, 0.4, 0.3, 0.2], 0.5) == 0.4  # two of four must pass
    assert pick_highest_floor([0.5, None, 0.4, 0.3], 0.5) == 0.4  # a missing score counts as no evidence
    assert pick_highest_floor([0.9] * 10, 0.9) == 0.9  # nine of ten, no float surprise
    with pytest.raises(ValueError):  # too many questions without a score
        pick_highest_floor([None, None, 0.5], 0.9)  # needs all three
    assert round_down(0.17899, 3) == 0.178 and round_down(0.178, 3) == 0.178  # down, and exact decimals stay


def test_leave_one_out_never_returns_the_pair_itself():  # own pair id removed from the candidates
    candidates = tuple(FakeScoredPair(pair, 1.0 - 0.1 * rank, rank) for rank, pair in enumerate(PAIRS))  # qa-101 first
    assert find_leave_one_out_neighbour(candidates, "qa-101").record.pair_id == "qa-102"  # skipped itself
    assert find_leave_one_out_neighbour(candidates[:1], "qa-101") is None  # only itself came back
    scores = score_leave_one_out(FixtureRetriever(FakeSearchSettings()), PAIRS, 0.5)  # every pair searched with its own question
    assert len(scores) == 4 and all(score.similarity == 0.0 and not score.is_correct for score in scores)  # fixture neighbours are unrelated, never the pair itself


def test_choose_pair_threshold_is_the_lowest_one_meeting_the_target():  # prefix scan, ties kept together
    scores = [NeighbourScore(0.99 - 0.01 * n, True) for n in range(60)] + [NeighbourScore(0.3, False)] * 40  # 60 correct above 0.4, 40 wrong at 0.3
    assert choose_pair_threshold(scores, 0.95, 50) == pytest.approx(0.40)  # lowest similarity of the clean prefix
    assert choose_pair_threshold(scores, 0.95, 70) is None  # only 60 are clean enough: below the support minimum no threshold qualifies
    assert measure_precision(scores, 0.3) == (0.6, 100) and measure_precision(scores, 2.0) == (None, 0)  # precision at a threshold


def test_pair_threshold_is_chosen_on_one_half_and_reported_on_the_other():  # DESIGN.md §11.4
    scores = [NeighbourScore(0.9 + 0.0001 * n, True) for n in range(200)] + [NeighbourScore(0.2, False)] * 200  # separable
    settings = dataclasses.replace(CalibrationSettings(), minimum_selected_pairs=20)  # small support for a small sample
    result = calibrate_pair_threshold(scores, settings, seed=1)  # seeded halves
    assert 0.9 <= result.threshold <= 0.92 and result.held_back_precision == 1.0  # clean separation on the held-back half too
    assert 0.4 <= result.held_back_coverage <= 0.6  # about half of the held-back pairs sit above it
    assert calibrate_pair_threshold([NeighbourScore(0.5, False)] * 100, settings, seed=1).threshold is None  # target out of reach


def test_pair_examples_on_fact_questions_are_judged_by_the_word_rules():  # fixture pairs answer fg-1..fg-3 but not fg-4
    retriever = FixtureRetriever(FakeSearchSettings())  # recorded pair rankings
    rules = [FactRule(golden_question.question, ((golden_question.accepted_terms[0],),), ()) for golden_question in load_fixture_golden_questions() if golden_question.answerable]  # fg-1..fg-4
    rule_book = FactRuleBook((), lambda answer, groups, rejected: all(any(term in answer.lower() for term in group) for group in groups), lambda term: None)  # judge
    assert judge_pair_examples_on_facts(retriever, rule_book, rules, 0.45) == (3, 3)  # three examples shown, all right; fg-4 has no similar pair
    assert sample_training_pairs(PAIRS, 3000, 1) != [] and len(sample_training_pairs(PAIRS, 2, 1)) == 2  # seeded sample, capped
