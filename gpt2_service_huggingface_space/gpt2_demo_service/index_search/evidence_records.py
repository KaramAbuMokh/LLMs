# What retrieval returns: scored passages and pairs, the route taken, the gate score and the time per stage
from dataclasses import dataclass  # frozen records
from enum import StrEnum  # routes as readable strings
from typing import Mapping  # stage timings
from retrieval_contracts.bundle_records import PassageRecord, QuestionAnswerRecord  # stored records

RETRIEVAL_STAGE_NAMES = ("encode", "dense_search", "lexical_search", "fusion", "rerank", "gate")  # keys of stage_milliseconds, in pipeline order


class EvidenceRoute(StrEnum):  # what retrieval found (or why it did not run)
    PASSAGES_AND_PAIRS = "passages_and_pairs"  # book passages and pair examples
    PASSAGES_ONLY = "passages_only"  # book passages only
    PAIRS_ONLY = "pairs_only"  # pair examples only
    NO_EVIDENCE = "no_evidence"  # searched, nothing good enough
    RETRIEVAL_OFF = "retrieval_off"  # disabled, unavailable or switched off by the request
    RETRIEVAL_FAILED = "retrieval_failed"  # an error inside this request's search


@dataclass(frozen=True)
class RankedRow:  # one row of a vector file or posting list with its score
    row: int  # row number inside the corpus
    score: float  # similarity or lexical weight (higher is better)


@dataclass(frozen=True)
class ScoredPassage:  # one candidate passage with every score it earned
    record: PassageRecord  # the stored passage
    fused_rank: int  # 1-based position after rank fusion, before any rerank
    dense_score: float  # exact cosine to the passage-search query (computed for every candidate, in every mode)
    lexical_score: float  # BM25 score when the passage is in the lexical candidate list, else 0.0
    rerank_score: float | None  # cross-encoder score, None when this passage was not reranked


@dataclass(frozen=True)
class ScoredPair:  # one candidate question-answer pair
    record: QuestionAnswerRecord  # the stored pair
    question_similarity: float  # exact cosine between the visitor question and the pair's question
    fused_rank: int  # 1-based position after rank fusion


@dataclass(frozen=True)
class RetrievedEvidence:  # the result of one retrieval
    route: EvidenceRoute  # what retrieval found
    passage_candidates: tuple[ScoredPassage, ...]  # ranked, before the gate, at most fused_candidate_count (the evaluation reads these)
    passages: tuple[ScoredPassage, ...]  # after the gate, at most max_passages, best first
    pair_candidates: tuple[ScoredPair, ...]  # ranked pair candidates before the threshold
    pair_examples: tuple[ScoredPair, ...]  # candidates at or above the threshold, at most max_pair_examples, best first
    best_evidence_score: float | None  # the gate score (best dense cosine, or best rerank score), None without candidates
    stage_milliseconds: Mapping[str, float]  # time per stage name in RETRIEVAL_STAGE_NAMES


def build_empty_evidence(route: EvidenceRoute) -> RetrievedEvidence:  # nothing found and no stages run (retrieval_off or retrieval_failed)
    return RetrievedEvidence(route, (), (), (), (), None, {})  # all empty


def choose_evidence_route(has_passages: bool, has_pair_examples: bool) -> EvidenceRoute:  # the route that names what a set of evidence holds
    if has_passages and has_pair_examples:  # both kinds
        return EvidenceRoute.PASSAGES_AND_PAIRS  # book passages and examples
    if has_passages:  # passages only
        return EvidenceRoute.PASSAGES_ONLY  # book passages
    return EvidenceRoute.PAIRS_ONLY if has_pair_examples else EvidenceRoute.NO_EVIDENCE  # examples, or nothing good enough
