# Test doubles for the service objects, driven by the shared fixture's expected rankings (no model, no bundle code of the service).
# They have the attribute names of DESIGN.md §7.5 and §7.6, so the evaluation runners cannot tell them from the real objects.
import dataclasses  # fake settings and fake records
from dataclasses import dataclass, field  # records
from typing import Any, Callable
from retrieval_contracts.bundle_manifest import PAIR_RECORDS_FILE_NAME, PASSAGE_RECORDS_FILE_NAME  # fixture bundle file names
from retrieval_contracts.bundle_records import PassageRecord, QuestionAnswerRecord, read_json_lines  # stored records
from retrieval_contracts.shared_fixture.fixture_files import FIXTURE_BUNDLE_FOLDER, load_fixture_expectations, load_fixture_golden_questions  # fixture


@dataclass(frozen=True)
class FakeSearchSettings:  # the fields of EvidenceSearchSettings that the evaluation reads or replaces
    passage_search_mode: str = "hybrid"  # hybrid, dense_only or lexical_only
    reranker_enabled: bool = False  # cross-encoder on or off
    fusion_constant: int = 60  # fusion constant
    max_passages: int = 3  # passages a prompt can hold
    passage_evidence_floor: float = 0.15  # dense gate
    rerank_evidence_floor: float = 0.0  # rerank gate
    pair_examples_enabled: bool = True  # pair search on (the calibration stage turns it on to see every pair)
    pair_example_threshold: float = 0.45  # pair gate


@dataclass(frozen=True)
class FakeScoredPassage:  # like ScoredPassage
    record: PassageRecord  # the stored passage
    fused_rank: int  # position in the candidate list


@dataclass(frozen=True)
class FakeScoredPair:  # like ScoredPair
    record: QuestionAnswerRecord  # the stored pair
    question_similarity: float  # cosine to the question
    fused_rank: int  # position in the list


@dataclass(frozen=True)
class FakeEvidence:  # like RetrievedEvidence
    passage_candidates: tuple[FakeScoredPassage, ...]  # ranked passages
    pair_candidates: tuple[FakeScoredPair, ...]  # ranked pairs
    best_evidence_score: float | None  # the gate score
    stage_milliseconds: dict[str, float]  # stage times


class FixtureRetriever:  # retrieve() answers from fixture_expectations.json; remembers every question it was asked
    def __init__(self, search_settings: FakeSearchSettings) -> None:  # one retriever per settings variant
        self.search_settings = search_settings  # the variant
        self.retrieved_questions: list[str] = []  # spy log
        self._expectations = load_fixture_expectations()["questions"]  # rankings by fixture question id
        self._ids_by_text = {question.question: question.question_id for question in load_fixture_golden_questions()}  # question text -> id
        self._passages = {record.passage_id: record for record in read_json_lines(FIXTURE_BUNDLE_FOLDER / PASSAGE_RECORDS_FILE_NAME, PassageRecord)}  # by id
        self._pairs = {record.pair_id: record for record in read_json_lines(FIXTURE_BUNDLE_FOLDER / PAIR_RECORDS_FILE_NAME, QuestionAnswerRecord)}  # by id

    def retrieve(self, question: str) -> FakeEvidence:  # ranked candidates for a fixture question
        self.retrieved_questions.append(question)  # spy
        if question not in self._ids_by_text:  # a stored pair's own question (leave-one-out calibration)
            own_first = sorted(self._pairs.values(), key=lambda pair: pair.question != question)  # the pair itself ranks first
            return FakeEvidence((), tuple(FakeScoredPair(pair, 1.0 if pair.question == question else 0.0, rank) for rank, pair in enumerate(own_first, start=1)),
                                None, {"encode": 1.0})  # no passages, similarity 1.0 for itself and 0.0 for the rest
        expected = self._expectations[self._ids_by_text[question]]  # recorded rankings
        dense_ids = [passage_id for passage_id, _ in expected["passage_dense_ranking"]]  # all passages by cosine
        lexical_ids = [passage_id for passage_id, _ in expected["passage_lexical_ranking"]]  # lexical hits
        fused_ids = [passage_id for passage_id, _ in expected["passage_fused_ranking"]]  # hybrid order
        mode = self.search_settings.passage_search_mode  # which search
        ids = {"hybrid": fused_ids, "dense_only": dense_ids, "lexical_only": lexical_ids + [i for i in dense_ids if i not in lexical_ids]}[mode]  # order
        scores = {"encode": 1.0, "dense_search": 2.0}  # stage times of every run
        best_score = expected["best_passage_dense_score"]  # dense gate score
        if self.search_settings.reranker_enabled:  # stand-in cross-encoder: shared-word count, stable for ties
            overlap = expected["passage_overlap_rerank_scores"]  # fixture rerank scores
            ids = sorted(ids, key=lambda passage_id: -overlap[passage_id])  # best overlap first
            best_score, scores["rerank"] = max(overlap.values()), 10.0  # rerank gate score and its time
        candidates = tuple(FakeScoredPassage(self._passages[passage_id], rank) for rank, passage_id in enumerate(ids, start=1))  # ranked passages
        pairs = tuple(FakeScoredPair(self._pairs[pair_id], similarity, rank) for rank, (pair_id, similarity) in enumerate(expected["pair_dense_ranking"], start=1))
        return FakeEvidence(candidates, pairs, best_score, scores)  # the evidence


@dataclass(frozen=True)
class FakeSource:  # like SourceCitation
    source_type: str  # "book_passage" or "question_answer_pair"
    book: str | None  # book title
    chapter: str | None  # chapter label
    snippet: str | None  # public snippet


@dataclass(frozen=True)
class FakeAnswer:  # like GroundedAnswer
    completion: str  # answer text
    retrieval_used: bool  # evidence in the prompt
    sources: tuple[FakeSource, ...]  # citations
    route: str  # route name
    prompt_token_count: int  # prompt length
    answer_token_count: int  # answer length
    stop_reason: str  # why it stopped
    verbatim_guard_cut: bool  # guard fired
    passage_ids: tuple[str, ...]  # packed passages
    pair_ids: tuple[str, ...]  # packed pairs
    best_evidence_score: float | None  # gate score


@dataclass
class FakeAnswerService:  # spy with the signature of GroundedAnswerService.answer_from_evidence
    completion_for: Callable[[str, str], str] = lambda question, layout: "an answer"  # (question, layout) -> completion text
    calls: list[tuple[str, int, str, int, Any]] = field(default_factory=list)  # (question, evidence object id, layout, budget, generator)

    def answer_from_evidence(self, answer_request: Any, evidence: FakeEvidence, layout: str, context_token_budget: int, random_generator: Any = None) -> FakeAnswer:  # records the call and answers from the evidence
        self.calls.append((answer_request.question, id(evidence), layout, context_token_budget, random_generator))  # spy
        uses_evidence = layout != "question_only" and bool(evidence.passage_candidates)  # the baseline ignores evidence
        passage_ids = (evidence.passage_candidates[0].record.passage_id,) if uses_evidence else ()  # first candidate packed
        return FakeAnswer(self.completion_for(answer_request.question, layout), uses_evidence, (), "passages_only" if uses_evidence else "no_evidence",
                          100 + context_token_budget // 10, 12, "end_token", False, passage_ids, (), evidence.best_evidence_score)


@dataclass(frozen=True)
class FakeAnswerRequest:  # like AnswerRequest (only the question matters to the fakes)
    question: str  # the question text


def replace_settings(settings: FakeSearchSettings, **changes: Any) -> FakeSearchSettings:  # dataclasses.replace under a test-friendly name
    return dataclasses.replace(settings, **changes)  # one changed field
