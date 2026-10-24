# Token F1, key-term recall and the invented-name detector
import pytest  # approximate comparisons
from retrieval_evaluation.metrics.answer_overlap_scoring import (  # under test
    find_capitalised_names, find_unsupported_names, key_term_recall, normalize_question_text, token_f1_score,
)


def test_token_f1_score():  # shared word multiset
    assert token_f1_score("the red keep", "the red keep") == 1.0  # identical
    assert token_f1_score("a b", "c d") == 0.0  # nothing shared
    assert token_f1_score("a b c d", "a b") == pytest.approx(2 * 0.5 * 1.0 / 1.5)  # precision 0.5, recall 1.0
    assert token_f1_score("", "") == 1.0 and token_f1_score("a", "") == 0.0  # empty texts


def test_normalize_question_text_ignores_case_and_punctuation():  # exact-match form
    assert normalize_question_text("Who is  the King's Hand?") == normalize_question_text("who is the king’s hand")  # curly apostrophe too


def test_key_term_recall_ignores_terms_the_question_already_has():  # only new information counts
    gold = "Gared believes that the dead have no business with the living."  # gold answer
    question = "What does Gared think about the dead?"  # shares gared and dead
    assert key_term_recall("They are not our business, the living matter", gold, question) == pytest.approx(2 / 3)  # business and living of believe, business, living
    assert key_term_recall("anything", "Gared.", "Who is Gared?") is None  # the gold answer adds nothing


def test_capitalised_names_skip_sentence_starts_that_are_function_words():  # names only
    assert find_capitalised_names("The Mountain serves. She met Tyrion's friend Bronn.") == ["Mountain", "Tyrion's", "Bronn"]  # The and She are skipped


def test_unsupported_names_find_the_invented_one():  # the Gregor-on-the-throne detector
    answer = "Gregor Clegane sits on the Iron Throne."  # a wrong answer
    assert find_unsupported_names(answer, ["Who sits on the Iron Throne?", "Joffrey sits the Iron Throne"]) == ["Gregor", "Clegane"]  # not in question or context
    assert find_unsupported_names("Joffrey sat there.", ["Joffrey's crown"]) == []  # possessive in the context supports the name
