# The `golden` stage: builds every question set from the real data, checks the fact-development set, writes results/golden_sets.json
from retrieval_contracts.rare_shingle_location import build_rare_shingle_index  # the shared answer locator
from retrieval_evaluation.evaluation_settings import EvaluationSettings  # paths and numbers
from retrieval_evaluation.experiments.evaluation_report_writing import read_json_results, write_json_results  # result files
from retrieval_evaluation.golden_sets.fact_dev_question_set import (  # the new fact questions and their three checks
    build_fact_development_questions, find_evidence_problems, find_overlap_problems, find_pair_duplicate_problems,
)
from retrieval_evaluation.golden_sets.fact_question_set import build_fact_questions, load_training_pair_questions  # G2
from retrieval_evaluation.golden_sets.golden_question_records import GoldenSets, golden_sets_from_dict, golden_sets_to_dict  # records
from retrieval_evaluation.golden_sets.held_out_passage_questions import build_held_out_split, load_answerable_validation_pairs, load_book_part_texts  # G1
from retrieval_evaluation.golden_sets.no_answer_questions import build_no_answer_questions  # G3
from retrieval_evaluation.golden_sets.regression_questions import build_regression_questions  # G4
from retrieval_evaluation.metrics.fact_rule_loading import FactRuleBook, load_fact_rule_book  # the 30 rules

GOLDEN_FILE_NAME = "golden_sets.json"  # inside the results folder


def load_rule_book(settings: EvaluationSettings) -> FactRuleBook:  # the Kaggle fact rules with their own judge
    return load_fact_rule_book(settings.locations.fact_rules_path, settings.locations.run_settings_path)  # from the two source files


def build_golden_sets(settings: EvaluationSettings) -> GoldenSets:  # builds and checks all question sets (needs about 1 GB for the locator)
    part_texts = load_book_part_texts(settings.locations.source_data_folder)  # seven part texts
    shingle_index = build_rare_shingle_index(part_texts)  # rare shingles of all parts (about 6 s)
    held_out = build_held_out_split(shingle_index, load_answerable_validation_pairs(settings.locations.source_data_folder), part_texts, settings.golden)  # G1
    del shingle_index  # frees the memory before the rest
    rule_book = load_rule_book(settings)  # the 30 rules
    pair_questions = load_training_pair_questions(settings.locations.source_data_folder)  # the 34,001 pair questions
    fact = build_fact_questions(rule_book, pair_questions)  # G2
    fact_development = build_fact_development_questions()  # FD
    problems = (find_overlap_problems(fact_development, fact, settings.golden.maximum_fact_overlap_f1)  # too close to one of the 30
                + find_pair_duplicate_problems(fact_development, pair_questions)  # word for word a pair question
                + find_evidence_problems(fact_development, rule_book, part_texts, settings.golden.evidence_window_characters))  # fact not where claimed
    if problems:  # a bad question would distort every choice made on it
        raise ValueError("fact-development set has problems: %s" % "; ".join(problems))  # ids and numbers only
    counts = {"held_out_answerable": held_out.answerable_count, "held_out_located": held_out.located_count,  # how G1 was cut
              "held_out_near_block_edge": held_out.near_edge_count, "held_out_development": len(held_out.development), "held_out_test": len(held_out.test),
              "fact_questions": len(fact), "fact_questions_word_for_word_in_pairs": sum(fact_question.is_word_for_word_in_pairs for fact_question in fact),
              "fact_development_questions_checked": len(fact_development), "no_answer_questions": len(build_no_answer_questions()),
              "regression_questions": len(build_regression_questions())}
    return GoldenSets(held_out.development, held_out.test, fact, fact_development, build_no_answer_questions(), build_regression_questions(), counts)  # the five sets plus counts


def run_golden_stage(settings: EvaluationSettings) -> GoldenSets:  # build, write, return
    golden_sets = build_golden_sets(settings)  # all sets
    write_json_results(settings.locations.results_folder / GOLDEN_FILE_NAME, golden_sets_to_dict(golden_sets))  # the record of what was used
    return golden_sets  # for the next stage


def load_golden_sets(settings: EvaluationSettings) -> GoldenSets:  # the sets of an earlier golden stage
    stored = read_json_results(settings.locations.results_folder / GOLDEN_FILE_NAME)  # None when never built
    if stored is None:  # the stage order matters
        raise FileNotFoundError("run the golden stage first (no %s in the results folder)" % GOLDEN_FILE_NAME)  # clear instruction
    return golden_sets_from_dict(stored)  # rebuilt records
