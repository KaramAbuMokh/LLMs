# FD rules (DESIGN.md §14.3): about 40 questions, none of the 30, token F1 <= 0.6 to each, evidence fields present; the checks catch violations
from dataclasses import replace  # tweak one record
from retrieval_evaluation.evaluation_settings import DataLocations  # source paths
from retrieval_evaluation.golden_sets.fact_dev_question_set import (  # under test
    build_fact_development_questions, find_evidence_problems, find_overlap_problems, find_pair_duplicate_problems,
)
from retrieval_evaluation.golden_sets.fact_question_set import build_fact_questions  # G2 records
from retrieval_evaluation.metrics.fact_rule_loading import load_fact_rule_book  # the 30 rules


def load_pieces():  # FD questions, G2 questions and the rule book
    locations = DataLocations()  # default paths
    rule_book = load_fact_rule_book(locations.fact_rules_path, locations.run_settings_path)  # the 30 rules
    return build_fact_development_questions(), build_fact_questions(rule_book, []), rule_book  # no pair questions needed here


def test_about_forty_questions_with_evidence_fields_and_unique_texts():  # size and shape
    questions, _, _ = load_pieces()  # loaded
    assert 38 <= len(questions) <= 45  # about 40
    assert len({question.rule.question for question in questions}) == len(questions)  # no repeats
    assert {question.evidence_part_number for question in questions} == {0, 1, 2, 3, 5}  # all five books
    assert all(question.evidence_character_offset > 0 and question.rule.accepted_groups for question in questions)  # evidence and rules present


def test_none_is_one_of_the_thirty_or_close_to_one():  # token F1 at most 0.6
    questions, fact_questions, _ = load_pieces()  # loaded
    assert find_overlap_problems(questions, fact_questions, 0.6) == []  # all clear
    copy = replace(questions[0], rule=replace(questions[0].rule, question=fact_questions[3].rule.question))  # a G2 question copied into FD
    assert find_overlap_problems([copy], fact_questions, 0.6) != []  # caught


def test_pair_duplicates_are_caught():  # word-for-word pair questions
    questions, _, _ = load_pieces()  # loaded
    assert find_pair_duplicate_problems(questions, ["unrelated question"]) == []  # clean
    assert find_pair_duplicate_problems(questions[:1], [questions[0].rule.question.upper().rstrip("?")]) != []  # normalised match


def test_evidence_check_needs_an_accepted_term_of_every_group_near_the_offset():  # on invented text
    questions, _, rule_book = load_pieces()  # loaded
    question = replace(questions[0], evidence_part_number=9, evidence_character_offset=100)  # points into a made-up part
    accepted_word = question.rule.accepted_groups[0][0].rstrip("*")  # a word the group accepts
    near = {9: "x " * 25 + accepted_word + " x" * 25}  # the word sits among other words, well inside the 2,000 window
    assert find_evidence_problems([question], rule_book, near, 2000) == []  # found
    assert find_evidence_problems([question], rule_book, {9: "nothing relevant here"}, 2000) != []  # term missing
    assert find_evidence_problems([question], rule_book, {}, 2000) != []  # part missing
    far = {9: accepted_word + " " + "x" * 5000}  # the word is far from a distant offset
    assert find_evidence_problems([replace(question, evidence_character_offset=4000)], rule_book, far, 2000) != []  # outside the window
