# Every number the evidence search consumes, declared WITHOUT defaults (the shipped values live in service_settings.py)
from dataclasses import dataclass  # frozen settings record
from typing import Literal  # the three search modes


@dataclass(frozen=True)
class EvidenceSearchSettings:  # how passages and pair examples are found, fused, gated and limited
    passage_search_mode: Literal["hybrid", "dense_only", "lexical_only"]  # which searches feed the passage ranking
    dense_candidate_count: int  # passages taken from the dense search
    lexical_candidate_count: int  # passages taken from the lexical search
    fusion_constant: int  # the k of reciprocal rank fusion (60)
    fused_candidate_count: int  # fused passages kept as candidates
    reranker_enabled: bool  # reorder the best candidates with the cross-encoder
    rerank_candidate_count: int  # how many fused candidates the cross-encoder reads
    passage_evidence_floor: float  # best dense cosine among the candidates must reach this (calibrated)
    rerank_evidence_floor: float  # best rerank score must reach this when the reranker is on (calibrated)
    max_passages: int  # passages handed to the prompt
    pair_examples_enabled: bool  # False skips the whole pair search (no candidates, no examples); True only for layouts that show pair examples
    pair_candidate_count: int  # pair candidates taken from each pair search
    pair_example_threshold: float  # question-to-question cosine a pair needs to become an example (calibrated; read only when pair_examples_enabled)
    max_pair_examples: int  # examples handed to the prompt
