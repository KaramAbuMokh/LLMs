# Majority scoring of fact questions and the held-out means
import pytest  # approximate comparisons
from retrieval_evaluation.metrics.answer_run_scoring import AnswerRecord, score_fact_records, score_held_out_records  # under test
from retrieval_evaluation.metrics.fact_rule_loading import FactRule, FactRuleBook  # rule types


def make_record(question_id: str, sample: int, completion: str, retrieval_used: bool = True, names: tuple = ()) -> AnswerRecord:  # one answer record
    return AnswerRecord(question_id, "set", "config", sample, completion, retrieval_used, "passages_only", (), (), 0.5, 100, 10, "end_token", False, 0, names, ())  # defaults


def make_rule_book() -> FactRuleBook:  # a judge that accepts answers containing the word "right"
    judge = lambda answer, accepted_groups, rejected_terms: "right" in answer  # stand-in for is_right
    return FactRuleBook((), judge, lambda term: None)  # no rules needed here


def test_two_of_three_samples_make_a_question_right():  # the Kaggle majority rule
    rule = FactRule("q", (("x",),), ())  # any rule
    records = [make_record("first_question", 0, "right"), make_record("first_question", 1, "right"), make_record("first_question", 2, "wrong"),  # 2 of 3
               make_record("second_question", 0, "right"), make_record("second_question", 1, "wrong"), make_record("second_question", 2, "wrong")]  # 1 of 3
    score = score_fact_records(records, make_rule_book(), {"first_question": rule, "second_question": rule})  # scored
    assert (score.question_count, score.majority_right, score.samples_right, score.sample_count) == (2, 1, 3, 6)  # counts
    assert score.right_question_ids == ("first_question",)  # which one


def test_held_out_means():  # recall, F1, invented names, no evidence
    references = {"first_question": ("Who is Gared?", "Gared guards the Wall."), "second_question": ("Who is Gared?", "Gared guards the Wall.")}  # question and gold answer
    records = [make_record("first_question", 0, "He guards the wall", True), make_record("second_question", 0, "Nobody", False, ("Gregor",))]  # one good, one invented
    score = score_held_out_records(records, references)  # scored
    assert score.key_term_recall == pytest.approx(0.5)  # guard and wall found by one answer, none by the other
    assert score.unsupported_name_rate == 0.5 and score.no_evidence_rate == 0.5  # one answer each
    assert score.mean_prompt_tokens == 100.0 and score.answer_count == 2  # lengths and count
