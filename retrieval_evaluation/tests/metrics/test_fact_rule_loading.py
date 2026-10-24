# The 30 fact rules come from the Kaggle source, so scoring is the function that produced 10/30
from retrieval_evaluation.evaluation_settings import DataLocations  # where the source files are
from retrieval_evaluation.metrics.fact_rule_loading import KNOWN_TEN_COUNT, load_fact_rule_book, load_module_from_path  # under test


def load_book():  # the rule book from the real source files (no book text involved)
    locations = DataLocations()  # default paths
    return load_fact_rule_book(locations.fact_rules_path, locations.run_settings_path), locations  # rule book and paths


def test_thirty_rules_and_the_first_ten_are_the_known_ten():  # DESIGN.md §14.3
    rule_book, locations = load_book()  # loaded
    known_ten = load_module_from_path("run_settings_for_test", locations.run_settings_path).FIXED_QUESTIONS  # the smoke questions
    assert len(rule_book.rules) == 30  # the fact set
    assert tuple(rule.question for rule in rule_book.rules[:KNOWN_TEN_COUNT]) == tuple(known_ten)  # the same ten, in order


def test_judge_is_the_sources_own_is_right():  # accepted groups and rejected terms
    rule_book, _ = load_book()  # loaded
    needle_rule = rule_book.rules[5]  # Arya's sword: needle, not longclaw
    assert rule_book.is_answer_right(needle_rule, "Her sword is called Needle.")  # right
    assert not rule_book.is_answer_right(needle_rule, "Needle and Longclaw.")  # rejected term present
    assert rule_book.term_occurs("carr* bran", "He carried Bran up the stairs")  # wildcard and phrase matching as in the source
    assert not rule_book.term_occurs("ned", "owned")  # whole words only


def test_loading_leaves_no_module_behind():  # sys.modules stays clean
    import sys  # module registry
    load_book()  # load once more
    assert "run_settings" not in sys.modules and "fact_questions" not in sys.modules  # nothing registered under the plain names
