# G2: the 30 fact questions of the Kaggle comparison (reported only), flagged for the known ten and for the three that are also stored pair questions
import json  # training pairs
from pathlib import Path  # source files
from typing import Iterable  # inputs
from retrieval_evaluation.golden_sets.golden_question_records import FactQuestion  # record
from retrieval_evaluation.metrics.answer_overlap_scoring import normalize_question_text  # comparable question form
from retrieval_evaluation.metrics.fact_rule_loading import KNOWN_TEN_COUNT, FactRuleBook  # the loaded rules

TRAINING_PAIRS_FILE_NAME = "question_answer_train.json"  # the 34,001 pairs the index holds


def load_training_pair_questions(source_data_folder: Path) -> list[str]:  # the question text of every training pair
    stored_pairs = json.loads((source_data_folder / TRAINING_PAIRS_FILE_NAME).read_text(encoding="utf-8"))  # [{"q", "a", ...}]
    return [stored["q"] for stored in stored_pairs]  # 34,001 questions


def build_fact_questions(rule_book: FactRuleBook, pair_questions: Iterable[str]) -> tuple[FactQuestion, ...]:  # G2 in source order
    stored_questions = {normalize_question_text(question) for question in pair_questions}  # comparable forms of the pair questions
    return tuple(  # one record per rule
        FactQuestion("g2-%02d" % number, rule, number <= KNOWN_TEN_COUNT, normalize_question_text(rule.question) in stored_questions)  # flags
        for number, rule in enumerate(rule_book.rules, start=1)  # numbered from 1
    )
