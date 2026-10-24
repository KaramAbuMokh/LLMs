# Calibration of the two thresholds (DESIGN.md §11.4): the evidence floor (highest floor that keeps enough G1 development and fact-development
# questions in) and the pair-example threshold (leave-one-out on training pairs: the pair itself is never its own neighbour).
import math  # rounding and share arithmetic
import random  # seeded sampling and halving
from dataclasses import dataclass  # frozen results
from typing import TYPE_CHECKING, Any, Sequence  # inputs
from retrieval_contracts.bundle_records import QuestionAnswerRecord  # a stored pair
from retrieval_evaluation.evaluation_settings import CalibrationSettings  # calibration numbers
from retrieval_evaluation.metrics.answer_overlap_scoring import token_f1_score  # answer similarity of a neighbour
from retrieval_evaluation.metrics.fact_rule_loading import FactRule, FactRuleBook  # word rules for fact questions

if TYPE_CHECKING:  # service type, for readers only
    from gpt2_demo_service.index_search.evidence_retriever import EvidenceRetriever


@dataclass(frozen=True)
class EvidenceFloorResult:  # the chosen floor and how each question set fares at it
    floor: float  # chosen passage evidence floor (rounded down)
    development_share: float  # G1 development questions that get evidence
    fact_development_share: float  # fact-development questions that get evidence (A7b)
    unanswerable_share: float  # G3 questions that still get evidence (A7)


@dataclass(frozen=True)
class NeighbourScore:  # the best other pair for one sampled pair
    similarity: float  # question-to-question cosine of the neighbour
    is_correct: bool  # the neighbour's answer has token F1 >= pair_correct_f1 with the pair's own answer


@dataclass(frozen=True)
class PairThresholdResult:  # the chosen threshold and its quality on the held-back half
    threshold: float | None  # None when no threshold reached the precision target
    chosen_half_precision: float | None  # precision on the half it was chosen on
    held_back_precision: float | None  # precision on the other half (A8)
    held_back_coverage: float  # share of the other half's pairs above the threshold


def share_with_evidence(scores: Sequence[float | None], floor: float) -> float:  # share of questions whose best score reaches the floor
    return sum(1 for score in scores if score is not None and score >= floor) / len(scores) if scores else 0.0  # None = no candidates


def pick_highest_floor(scores: Sequence[float | None], minimum_share: float) -> float:  # highest floor at which minimum_share of the questions still pass
    needed_count = max(1, math.ceil(round(minimum_share * len(scores), 9)))  # questions that must pass
    ordered = sorted((score if score is not None else float("-inf") for score in scores), reverse=True)  # best first, missing last
    if ordered[needed_count - 1] == float("-inf"):  # too many questions have no score at all
        raise ValueError("fewer than %d questions have an evidence score" % needed_count)  # the retriever returned nothing
    return ordered[needed_count - 1]  # the needed_count-th best score is the highest floor that keeps them


def round_down(value: float, digits: int) -> float:  # rounds toward minus infinity so coverage never falls below the requirement
    return math.floor(round(value * 10 ** digits, 9)) / 10 ** digits  # exact decimals


def calibrate_evidence_floor(development_scores: Sequence[float | None], fact_development_scores: Sequence[float | None],  # the evidence floor and how each set fares
                             unanswerable_scores: Sequence[float | None], settings: CalibrationSettings) -> EvidenceFloorResult:
    floor = min(pick_highest_floor(development_scores, settings.development_evidence_share),  # tight for G1 development
                pick_highest_floor(fact_development_scores, settings.fact_development_evidence_share))  # and for general questions
    floor = round_down(floor, settings.floor_rounding_digits)  # readable value, never stricter than required
    return EvidenceFloorResult(  # how each set fares
        floor, share_with_evidence(development_scores, floor), share_with_evidence(fact_development_scores, floor),
        share_with_evidence(unanswerable_scores, floor),
    )


def build_floor_tradeoff(development_scores: Sequence[float | None], fact_development_scores: Sequence[float | None],  # one row per development coverage target
                         unanswerable_scores: Sequence[float | None], development_shares: Sequence[float], digits: int) -> list[dict[str, float]]:
    rows = []  # one row per development coverage target
    for share in development_shares:  # e.g. 1.0, 0.9, 0.85, ...
        floor = round_down(pick_highest_floor(development_scores, share), digits)  # highest floor that keeps that share of development questions
        rows.append({"target_development_share": share, "floor": floor, "development_share": share_with_evidence(development_scores, floor),  # what each set gets
                     "fact_development_share": share_with_evidence(fact_development_scores, floor), "unanswerable_share": share_with_evidence(unanswerable_scores, floor)})
    return rows  # shows whether A7 (few false alarms) and coverage can both hold


def find_leave_one_out_neighbour(pair_candidates: Sequence[Any], own_pair_id: str) -> Any | None:  # first candidate that is not the pair itself
    return next((candidate for candidate in pair_candidates if candidate.record.pair_id != own_pair_id), None)  # None when only itself came back


def sample_training_pairs(pairs: Sequence[QuestionAnswerRecord], sample_size: int, seed: int) -> list[QuestionAnswerRecord]:  # seeded sample
    return random.Random(seed).sample(list(pairs), min(sample_size, len(pairs)))  # reproducible


def score_leave_one_out(retriever: "EvidenceRetriever", sampled_pairs: Sequence[QuestionAnswerRecord], correct_f1: float) -> list[NeighbourScore]:
    """The retriever must show every pair candidate (its pair_example_threshold set to -1 by the caller)."""
    scores = []  # one per pair that has a neighbour
    for pair in sampled_pairs:  # every sampled pair
        neighbour = find_leave_one_out_neighbour(retriever.retrieve(pair.question).pair_candidates, pair.pair_id)  # its own id removed
        if neighbour is not None:  # at least one other candidate
            scores.append(NeighbourScore(neighbour.question_similarity, token_f1_score(neighbour.record.answer, pair.answer) >= correct_f1))  # judged
    return scores  # in sample order


def measure_precision(scores: Sequence[NeighbourScore], threshold: float) -> tuple[float | None, int]:  # (precision, selected count) at a threshold
    selected = [score for score in scores if score.similarity >= threshold]  # neighbours that would become examples
    return (sum(score.is_correct for score in selected) / len(selected) if selected else None), len(selected)  # None when nothing is selected


def choose_pair_threshold(scores: Sequence[NeighbourScore], precision_target: float, minimum_selected: int) -> float | None:
    """The lowest threshold whose selection has precision >= precision_target and at least minimum_selected pairs."""
    ordered = sorted(scores, key=lambda score: score.similarity, reverse=True)  # most similar first
    correct_so_far, chosen = 0, None  # running count and best threshold so far
    for position, score in enumerate(ordered):  # grow the selection one pair at a time
        correct_so_far += score.is_correct  # count it
        is_prefix_end = position == len(ordered) - 1 or ordered[position + 1].similarity != score.similarity  # a threshold never splits equal scores
        if is_prefix_end and position + 1 >= minimum_selected and correct_so_far / (position + 1) >= precision_target:  # qualifies
            chosen = score.similarity  # keep going: a lower qualifying threshold covers more pairs
    return chosen  # None when no prefix qualifies


def calibrate_pair_threshold(scores: Sequence[NeighbourScore], settings: CalibrationSettings, seed: int) -> PairThresholdResult:  # choose on one half, report on the other
    shuffled = list(scores)  # work on a copy
    random.Random(seed).shuffle(shuffled)  # seeded shuffle, then two halves
    middle = len(shuffled) // 2  # cut point
    chosen_half, held_back_half = shuffled[:middle], shuffled[middle:]  # choose on one, report on the other
    raw_threshold = choose_pair_threshold(chosen_half, settings.pair_precision_target, settings.minimum_selected_pairs)  # unrounded choice
    if raw_threshold is None:  # the target is out of reach
        return PairThresholdResult(None, None, None, 0.0)  # no pair examples
    factor = 10 ** settings.floor_rounding_digits  # decimals kept
    threshold = math.ceil(round(raw_threshold * factor, 9)) / factor  # rounded UP: stricter, so precision cannot fall below the measured one
    held_back_precision, selected_count = measure_precision(held_back_half, threshold)  # A8
    return PairThresholdResult(  # the report
        threshold, measure_precision(chosen_half, threshold)[0], held_back_precision, selected_count / len(held_back_half) if held_back_half else 0.0,
    )


def judge_pair_examples_on_facts(retriever: "EvidenceRetriever", rule_book: FactRuleBook, rules: Sequence[FactRule],
                                 threshold: float) -> tuple[int, int]:  # (examples shown, examples whose answer satisfies the question's rule)
    shown, correct = 0, 0  # counters
    for rule in rules:  # every fact-development question
        candidates = retriever.retrieve(rule.question).pair_candidates  # pair candidates, best first
        if candidates and candidates[0].question_similarity >= threshold:  # the top one would become an example
            shown += 1  # counted
            correct += rule_book.is_answer_right(rule, candidates[0].record.answer)  # judged by the same word rules
    return shown, correct  # a general-question check of the threshold
