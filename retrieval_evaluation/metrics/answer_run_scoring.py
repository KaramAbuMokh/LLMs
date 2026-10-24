# Scores for one configuration of an answer run: the fact majority, and the held-out means (key-term recall, F1, invented names, no evidence)
from dataclasses import dataclass  # frozen records
from typing import Mapping, Sequence  # inputs
from retrieval_evaluation.metrics.answer_overlap_scoring import key_term_recall, token_f1_score  # answer overlap measures
from retrieval_evaluation.metrics.fact_rule_loading import FactRule, FactRuleBook  # the word rules and their judge
from retrieval_evaluation.metrics.ranking_metrics import average_values  # mean of per-question values


@dataclass(frozen=True)
class AnswerRecord:  # one generated answer: ids, scores and the model's own words, never passage text
    question_id: str  # question set id
    set_name: str  # "fact_development", "held_out_development", "held_out_test", "fact", "regression"
    configuration_name: str  # layout and budget, e.g. "passages_and_pair_examples_600"
    sample: int  # sample number inside the question
    completion: str  # the answer after the verbatim guard
    retrieval_used: bool  # at least one passage or pair is in the prompt
    route: str  # what the packed prompt holds
    passage_ids: tuple[str, ...]  # passages in the prompt
    pair_ids: tuple[str, ...]  # pairs in the prompt
    best_evidence_score: float | None  # gate score of the retrieval
    prompt_token_count: int  # tokens of the packed prompt
    answer_token_count: int  # tokens generated
    stop_reason: str  # why generation ended
    verbatim_guard_cut: bool  # the runtime guard shortened the answer
    book_snippet_count: int  # book snippets among the sources
    unsupported_names: tuple[str, ...]  # names in the answer found in neither the question nor the packed context
    copyright_problems: tuple[str, ...]  # A16 findings for the response (numbers only)


@dataclass(frozen=True)
class FactSetScore:  # the facts score of one configuration on one fact question set
    question_count: int  # questions scored
    majority_right: int  # questions with more than half of their samples right
    samples_right: int  # right samples
    sample_count: int  # all samples
    right_question_ids: tuple[str, ...]  # ids of the questions with a right majority


@dataclass(frozen=True)
class HeldOutScore:  # the means of one configuration on held-out passage questions
    answer_count: int  # answers scored
    key_term_recall: float  # mean share of the gold answer's new key terms found
    token_f1: float  # mean token F1 against the gold answer
    unsupported_name_rate: float  # share of answers naming something not in the question or the context
    no_evidence_rate: float  # share of answers whose prompt held no passage and no pair
    mean_prompt_tokens: float  # mean packed prompt length


def group_by_question(records: Sequence[AnswerRecord]) -> dict[str, list[AnswerRecord]]:  # samples of each question together
    grouped: dict[str, list[AnswerRecord]] = {}  # question id -> its samples
    for record in records:  # every record
        grouped.setdefault(record.question_id, []).append(record)  # add to its question
    return grouped  # in first-seen order


def score_fact_records(records: Sequence[AnswerRecord], rule_book: FactRuleBook, rules_by_question_id: Mapping[str, FactRule]) -> FactSetScore:
    """Majority rule as in the Kaggle scoring: a question is right when more than half of its samples satisfy the rule."""
    grouped = group_by_question(records)  # samples per question
    right_ids, samples_right = [], 0  # outcome so far
    for question_id, samples in grouped.items():  # every question
        right_count = sum(rule_book.is_answer_right(rules_by_question_id[question_id], sample.completion) for sample in samples)  # right samples
        samples_right += right_count  # count them
        if right_count * 2 > len(samples):  # 2 of 3
            right_ids.append(question_id)  # right by majority
    return FactSetScore(len(grouped), len(right_ids), samples_right, len(records), tuple(right_ids))  # the score


def score_held_out_records(records: Sequence[AnswerRecord], reference_by_question_id: Mapping[str, tuple[str, str]]) -> HeldOutScore:
    """reference_by_question_id maps a question id to (question, gold answer)."""
    recalls, f1_scores = [], []  # per-answer values
    for record in records:  # every answer
        question, gold_answer = reference_by_question_id[record.question_id]  # the reference
        recall = key_term_recall(record.completion, gold_answer, question)  # None when the gold answer adds no new term
        if recall is not None:  # left out of the mean otherwise
            recalls.append(recall)  # keep
        f1_scores.append(token_f1_score(record.completion, gold_answer))  # always defined
    return HeldOutScore(  # the means
        len(records), average_values(recalls), average_values(f1_scores),
        average_values([1.0 if record.unsupported_names else 0.0 for record in records]),  # answers with an invented name
        average_values([0.0 if record.retrieval_used else 1.0 for record in records]),  # answers without any evidence
        average_values([float(record.prompt_token_count) for record in records]),  # prompt length
    )
