# G2 flags, G3 and G4 shapes, and the JSON round trip of the golden sets
import json  # dumps and loads
from retrieval_evaluation.evaluation_settings import DataLocations  # source paths
from retrieval_evaluation.golden_sets.fact_question_set import build_fact_questions  # G2
from retrieval_evaluation.golden_sets.fact_dev_question_set import build_fact_development_questions  # FD
from retrieval_evaluation.golden_sets.golden_question_records import ABSENT_KIND, UNRELATED_KIND, GoldenSets, HeldOutQuestion  # records
from retrieval_evaluation.golden_sets.golden_question_records import golden_sets_from_dict, golden_sets_to_dict  # round trip
from retrieval_evaluation.golden_sets.no_answer_questions import build_no_answer_questions  # G3
from retrieval_evaluation.golden_sets.regression_questions import build_regression_questions, is_iron_throne_question  # G4
from retrieval_contracts.rare_shingle_location import ShingleHit  # a gold hit
from retrieval_evaluation.metrics.fact_rule_loading import load_fact_rule_book  # the 30 rules


def test_fact_set_flags_the_known_ten_and_word_for_word_pair_questions():  # A2 and A3 inputs
    locations = DataLocations()  # default paths
    rule_book = load_fact_rule_book(locations.fact_rules_path, locations.run_settings_path)  # the rules
    facts = build_fact_questions(rule_book, ["What is the name of Arya Stark's sword?", "Something else"])  # one pair question matches
    assert sum(question.is_known_ten for question in facts) == 10 and facts[9].is_known_ten and not facts[10].is_known_ten  # first ten
    assert [question.question_id for question in facts if question.is_word_for_word_in_pairs] == ["g2-06"]  # Arya's sword


def test_no_answer_set_has_forty_questions_in_two_kinds():  # 25 off-topic, 15 absent
    questions = build_no_answer_questions()  # built
    assert len(questions) == 40 and len({question.question_id for question in questions}) == 40  # unique ids
    assert sum(question.kind == UNRELATED_KIND for question in questions) == 25 and sum(question.kind == ABSENT_KIND for question in questions) == 15  # kinds


def test_regression_set_has_the_four_questions_and_two_iron_throne_ones():  # G4
    questions = build_regression_questions()  # built
    assert len(questions) == 4 and sum(is_iron_throne_question(question) for question in questions) == 2  # two Iron Throne questions
    assert all("mountain" in question.rule.rejected_terms for question in questions if is_iron_throne_question(question))  # the old wrong answer is rejected


def test_golden_sets_survive_a_json_round_trip():  # what golden_sets.json stores is what the next stage loads
    locations = DataLocations()  # default paths
    rule_book = load_fact_rule_book(locations.fact_rules_path, locations.run_settings_path)  # the rules
    held_out = (HeldOutQuestion("g1-1", "Who?", "Him.", (ShingleHit(6, 10, 20), ShingleHit(6, 30, 40)), 25, "development"),)  # one G1 question
    golden = GoldenSets(held_out, (), build_fact_questions(rule_book, []), build_fact_development_questions(), build_no_answer_questions(),
                        build_regression_questions(), {"held_out_development": 1})  # all sets
    restored = golden_sets_from_dict(json.loads(json.dumps(golden_sets_to_dict(golden))))  # through real JSON text
    assert restored == golden  # identical records
