# Finds evidence for one question: hybrid search over book passages and over stored question-answer pairs, an optional rerank and one gate
from dataclasses import replace  # copies a record with one field changed
import numpy  # arrays
from numpy.typing import NDArray  # array type hints
from gpt2_demo_service.index_loading.bundle_loading import LoadedIndexBundle  # the validated bundle
from gpt2_demo_service.index_search.dense_search import find_top_dense_matches  # exact cosine search
from gpt2_demo_service.index_search.evidence_records import RetrievedEvidence, ScoredPair, ScoredPassage, choose_evidence_route  # result records
from gpt2_demo_service.index_search.evidence_search_settings import EvidenceSearchSettings  # every number the search uses
from gpt2_demo_service.index_search.lexical_search import find_top_lexical_matches  # BM25 search
from gpt2_demo_service.index_search.rank_fusion import fuse_rankings  # reciprocal rank fusion
from gpt2_demo_service.monitoring.stage_timing import StageTimer  # time per stage
from retrieval_contracts.encoder_contracts import PassagePairScorer, TextEncoder  # the two encoder interfaces
from retrieval_contracts.indexed_text import compose_indexed_passage_text  # the text the reranker reads


class EvidenceRetriever:  # read-only after construction; the service calls retrieve() inside its single inference lock
    def __init__(self, bundle: LoadedIndexBundle, text_encoder: TextEncoder, settings: EvidenceSearchSettings,  # wires the bundle, the encoders and the numbers
                 pair_scorer: PassagePairScorer | None = None) -> None:  # the optional reranker defaults to none
        if settings.reranker_enabled and pair_scorer is None:  # the switch needs its model
            raise ValueError("the reranker is enabled but no pair scorer was given")  # a wiring bug, caught at startup
        self._bundle = bundle  # passages, pairs and their indexes
        self._text_encoder = text_encoder  # embeds the question
        self._settings = settings  # all numbers
        self._pair_scorer = pair_scorer  # the optional reranker
        self._passage_rows = {record.passage_id: row for row, record in enumerate(bundle.passages)}  # passage id -> row
        self._pair_rows = {record.pair_id: row for row, record in enumerate(bundle.pairs)}  # pair id -> row

    def retrieve(self, question: str) -> RetrievedEvidence:  # passages and pair examples for one visitor question
        timer = StageTimer()  # stage clock
        instruction = self._text_encoder.fingerprint.passage_query_instruction  # prefix of passage-search questions
        with timer.measure("encode"):  # one batch: the question alone (pairs) and with the instruction (passages)
            question_vector, passage_query_vector = self._text_encoder.encode_texts([question, instruction + question])  # row 0 = question alone, row 1 = question with the instruction
        candidates = self._rank_passages(question, passage_query_vector, timer)  # fused passage candidates
        if self._settings.reranker_enabled:  # the cross-encoder reorders the best candidates
            candidates = self._rerank_passages(question, candidates, timer)  # reorder the head with the cross-encoder
        pair_candidates = self._rank_pairs(question, question_vector, timer) if self._settings.pair_examples_enabled else ()  # fused pair candidates; no search at all when pairs are off
        with timer.measure("gate"):  # decide what reaches the prompt
            best_score, passages = self._apply_evidence_gate(candidates)  # one calibrated score decides
            pair_examples = tuple(pair for pair in pair_candidates if pair.question_similarity >= self._settings.pair_example_threshold)  # candidates close enough to become examples (none when pairs are off)
        pair_examples = pair_examples[:self._settings.max_pair_examples]  # at most this many examples
        return RetrievedEvidence(  # everything the prompt packer and the evaluation need
            choose_evidence_route(bool(passages), bool(pair_examples)), tuple(candidates), passages, pair_candidates, pair_examples, best_score,  # route, candidates, gated passages, pair candidates, pair examples, gate score
            timer.stage_milliseconds,  # stage timings
        )

    def _rank_passages(self, question: str, passage_query_vector: NDArray[numpy.float32], timer: StageTimer) -> list[ScoredPassage]:  # dense + lexical search and fusion over the passages
        settings, bundle = self._settings, self._bundle  # short names for this method
        ranked_id_lists, lexical_scores = [], {}  # one id list per search that runs, BM25 score per row
        if settings.passage_search_mode != "lexical_only":  # dense search runs
            with timer.measure("dense_search"):  # exact cosine over every passage
                dense_rows = find_top_dense_matches(bundle.passage_vectors, passage_query_vector, settings.dense_candidate_count)  # exact cosine ranking of the passages
            ranked_id_lists.append([bundle.passages[ranked.row].passage_id for ranked in dense_rows])  # dense ranking
        if settings.passage_search_mode != "dense_only":  # lexical search runs
            with timer.measure("lexical_search"):  # BM25 over the heading + text
                lexical_rows = find_top_lexical_matches(bundle.passage_lexical, question, settings.lexical_candidate_count)  # BM25 ranking of the passages
            ranked_id_lists.append([bundle.passages[ranked.row].passage_id for ranked in lexical_rows])  # lexical ranking
            lexical_scores = {ranked.row: ranked.score for ranked in lexical_rows}  # kept for the candidates' records
        with timer.measure("fusion"):  # merge the rankings
            fused = fuse_rankings(ranked_id_lists, settings.fusion_constant)[:settings.fused_candidate_count]  # fused ranking cut to the candidate count
        return [  # candidates with exact cosines, whatever search found them
            ScoredPassage(bundle.passages[row], rank, float(bundle.passage_vectors[row] @ passage_query_vector), lexical_scores.get(row, 0.0), None)  # one record per fused candidate, scores computed exactly
            for rank, row in enumerate((self._passage_rows[candidate.item_id] for candidate in fused), start=1)  # rows of the fused candidates, ranked from 1
        ]

    def _rerank_passages(self, question: str, candidates: list[ScoredPassage], timer: StageTimer) -> list[ScoredPassage]:  # cross-encoder reorders the best fused candidates
        head = candidates[:self._settings.rerank_candidate_count]  # only the best fused candidates are read
        with timer.measure("rerank"):  # cross-encoder over (question, heading + text)
            scores = self._pair_scorer.score_pairs(question, [compose_indexed_passage_text(candidate.record) for candidate in head])  # one score per head candidate
        scored_head = [replace(candidate, rerank_score=float(score)) for candidate, score in zip(head, scores)]  # scores attached
        scored_head.sort(key=lambda candidate: -candidate.rerank_score)  # stable: equal scores keep the fused order
        return scored_head + candidates[self._settings.rerank_candidate_count:]  # the rest stays behind in fused order

    def _apply_evidence_gate(self, candidates: list[ScoredPassage]) -> tuple[float | None, tuple[ScoredPassage, ...]]:  # decides whether the candidates may reach the prompt
        if not candidates:  # the corpus returned nothing
            return None, ()  # no score, no passages
        if self._settings.reranker_enabled:  # the reranked head decides
            best_score = max(candidate.rerank_score for candidate in candidates if candidate.rerank_score is not None)  # best rerank score
            floor = self._settings.rerank_evidence_floor  # calibrated for rerank scores
        else:  # dense cosine decides
            best_score = max(candidate.dense_score for candidate in candidates)  # best cosine among the candidates
            floor = self._settings.passage_evidence_floor  # calibrated for cosines
        passages = tuple(candidates[:self._settings.max_passages]) if best_score >= floor else ()  # evidence only at or above the floor
        return best_score, passages  # the gate score and what passed

    def _rank_pairs(self, question: str, question_vector: NDArray[numpy.float32], timer: StageTimer) -> tuple[ScoredPair, ...]:  # dense + lexical search and fusion over the stored pairs
        settings, bundle = self._settings, self._bundle  # short names for this method
        with timer.measure("dense_search"):  # question-to-question cosine
            dense_rows = find_top_dense_matches(bundle.question_vectors, question_vector, settings.pair_candidate_count)  # question-to-question cosine ranking
        with timer.measure("lexical_search"):  # BM25 over question + answer
            lexical_rows = find_top_lexical_matches(bundle.pair_lexical, question, settings.pair_candidate_count)  # BM25 ranking over question + answer
        with timer.measure("fusion"):  # merge the two pair rankings
            fused = fuse_rankings([[bundle.pairs[ranked.row].pair_id for ranked in rows] for rows in (dense_rows, lexical_rows)], settings.fusion_constant)  # fuse the two pair rankings by id
        return tuple(  # exact question similarities for the kept candidates
            ScoredPair(bundle.pairs[row], float(bundle.question_vectors[row] @ question_vector), rank)  # one record per kept pair with its exact question similarity
            for rank, row in enumerate((self._pair_rows[candidate.item_id] for candidate in fused[:settings.pair_candidate_count]), start=1)  # rows of the fused pairs, ranked from 1
        )
