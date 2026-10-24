# The question records of the five golden sets and their JSON form (results/golden_sets.json). Questions, gold offsets and counts only: no book text.
import dataclasses  # nested dataclass to dictionary
from dataclasses import dataclass  # frozen records
from typing import Any, Mapping  # JSON-shaped values
from retrieval_contracts.rare_shingle_location import ShingleHit  # a gold hit: a character span of one part text
from retrieval_evaluation.metrics.fact_rule_loading import FactRule  # question plus word rules

DEVELOPMENT_SPLIT = "development"  # G1 questions in an even block: used to choose settings
TEST_SPLIT = "test"  # G1 questions in an odd block: reported
UNRELATED_KIND = "off_topic"  # G3 question that has nothing to do with the books
ABSENT_KIND = "not_in_books"  # G3 question that sounds like the series but the books do not contain


@dataclass(frozen=True)
class HeldOutQuestion:  # G1: one answerable held-out question with its gold hits in part 6
    question_id: str  # "g1-<source index>"
    question: str  # the question text
    gold_answer: str  # the reference answer written with the question
    gold_hits: tuple[ShingleHit, ...]  # rare-shingle hits of question + answer (never widened)
    gold_center: int  # (first hit start + last hit end) // 2: decides the block
    split: str  # DEVELOPMENT_SPLIT or TEST_SPLIT


@dataclass(frozen=True)
class FactQuestion:  # G2: one of the 30 fact questions (reported only)
    question_id: str  # "g2-<number>"
    rule: FactRule  # question and word rules from the Kaggle source
    is_known_ten: bool  # one of the first ten (A3)
    is_word_for_word_in_pairs: bool  # the question text is also a stored pair question (A2 leaves these three out)


@dataclass(frozen=True)
class FactDevelopmentQuestion:  # FD: a new fact question used to choose settings
    question_id: str  # "fd-<number>"
    rule: FactRule  # question and word rules in the same format
    evidence_part_number: int  # part where the fact is stated
    evidence_character_offset: int  # offset near which an accepted term of each group appears (no book text stored)


@dataclass(frozen=True)
class NoAnswerQuestion:  # G3: a question the books cannot answer
    question_id: str  # "g3-<number>"
    question: str  # the question text
    kind: str  # UNRELATED_KIND or ABSENT_KIND


@dataclass(frozen=True)
class RegressionQuestion:  # G4: a question the old model got wrong (reported only)
    question_id: str  # "g4-<number>"
    rule: FactRule  # question and word rules


@dataclass(frozen=True)
class GoldenSets:  # every question set plus the counts that explain how they were cut
    held_out_development: tuple[HeldOutQuestion, ...]  # G1 development
    held_out_test: tuple[HeldOutQuestion, ...]  # G1 test
    fact: tuple[FactQuestion, ...]  # G2
    fact_development: tuple[FactDevelopmentQuestion, ...]  # FD
    no_answer: tuple[NoAnswerQuestion, ...]  # G3
    regression: tuple[RegressionQuestion, ...]  # G4
    counts: Mapping[str, int]  # answerable, located, near-edge, development, test, ...


def build_fact_rule(values: Mapping[str, Any]) -> FactRule:  # rebuilds a rule from its JSON form
    return FactRule(values["question"], tuple(tuple(group) for group in values["accepted_groups"]), tuple(values["rejected_terms"]))  # tuples again


def build_held_out_question(values: Mapping[str, Any]) -> HeldOutQuestion:  # rebuilds a G1 question from its JSON form
    return HeldOutQuestion(  # frozen record
        values["question_id"], values["question"], values["gold_answer"], tuple(ShingleHit(**hit) for hit in values["gold_hits"]),  # identity, hits
        values["gold_center"], values["split"],  # block decision
    )


def golden_sets_to_dict(golden_sets: GoldenSets) -> dict[str, Any]:  # JSON-ready dictionary (tuples become lists when dumped)
    return dataclasses.asdict(golden_sets)  # nested records follow


def golden_sets_from_dict(values: Mapping[str, Any]) -> GoldenSets:  # the inverse of golden_sets_to_dict
    return GoldenSets(  # frozen aggregate
        tuple(build_held_out_question(item) for item in values["held_out_development"]),  # G1 development
        tuple(build_held_out_question(item) for item in values["held_out_test"]),  # G1 test
        tuple(FactQuestion(i["question_id"], build_fact_rule(i["rule"]), i["is_known_ten"], i["is_word_for_word_in_pairs"]) for i in values["fact"]),  # G2
        tuple(FactDevelopmentQuestion(i["question_id"], build_fact_rule(i["rule"]), i["evidence_part_number"], i["evidence_character_offset"])
              for i in values["fact_development"]),  # FD
        tuple(NoAnswerQuestion(**item) for item in values["no_answer"]),  # G3
        tuple(RegressionQuestion(i["question_id"], build_fact_rule(i["rule"])) for i in values["regression"]),  # G4
        dict(values["counts"]),  # counts
    )
