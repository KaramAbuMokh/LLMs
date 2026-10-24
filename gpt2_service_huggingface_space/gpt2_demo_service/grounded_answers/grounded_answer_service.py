# The one answer path: retrieve (or skip), pack the prompt, generate, guard against copying, cite the sources; the service and the evaluation share it
import logging  # error line when one retrieval fails
from dataclasses import dataclass  # frozen records
from typing import Mapping  # stage timings
import torch  # seeded random generators
from gpt2_demo_service.grounded_answers.answer_settings import GroundedAnswerSettings, PromptLayout  # settings and layout names
from gpt2_demo_service.grounded_answers.book_run_index import BookRunIndex  # every 26-word run of the books
from gpt2_demo_service.grounded_answers.context_packing import PackedPrompt, PromptBudget, count_tokens, pack_prompt  # token-budgeted prompt packing
from gpt2_demo_service.grounded_answers.prompt_layouts import render_question_block  # the trained question template
from gpt2_demo_service.grounded_answers.source_citations import SourceCitation, build_source_citations  # public citations
from gpt2_demo_service.grounded_answers.verbatim_guard import limit_verbatim_overlap  # copy guard
from gpt2_demo_service.index_search.evidence_records import EvidenceRoute, RetrievedEvidence, build_empty_evidence, choose_evidence_route  # retrieval results
from gpt2_demo_service.index_search.evidence_retriever import EvidenceRetriever  # the search
from gpt2_demo_service.language_models.got_text_generator import GotTextGenerator, SamplingSettings, StopReason  # the text model
from gpt2_demo_service.monitoring.stage_timing import StageTimer  # time per stage

logger = logging.getLogger(__name__)  # logger for this module


class QuestionTooLongError(ValueError):  # the question block alone leaves no room to answer inside the model's context (a visitor input problem, not a bug)
    def __init__(self, max_question_tokens: int) -> None:  # carries the limit for the caller's message
        super().__init__("questions can be up to %d tokens" % max_question_tokens)  # readable text
        self.max_question_tokens = max_question_tokens  # tokens of the question itself that still fit


@dataclass(frozen=True)  # immutable record
class AnswerRequest:  # one question as the answer path sees it
    question: str  # the visitor's question
    max_new_tokens: int  # tokens the visitor asked for (the reserve is clamped to the settings' floor and ceiling)
    sampling: SamplingSettings  # how words are chosen
    stop_strings: tuple[str, ...]  # extra end texts from the request
    use_retrieval: bool  # False asks for today's plain prompt


@dataclass(frozen=True)  # immutable record
class GroundedAnswer:  # what the visitor gets and what the log and the evaluation record
    completion: str  # the answer text
    retrieval_used: bool  # True exactly when a passage or pair example is in the prompt
    sources: tuple[SourceCitation, ...]  # citations of the packed evidence
    route: EvidenceRoute  # retrieval_off/failed as found, otherwise what the packed prompt holds
    prompt_token_count: int  # tokens in the prompt
    answer_token_count: int  # tokens generated
    stop_reason: StopReason  # why generation ended
    stage_milliseconds: Mapping[str, float]  # retrieval stages plus packing, prefill and decode
    verbatim_guard_cut: bool  # the copy guard shortened the answer
    passage_ids: tuple[str, ...]  # packed passages (for the log line and the evaluation)
    pair_ids: tuple[str, ...]  # packed pair examples
    best_evidence_score: float | None  # the gate score of this retrieval


class GroundedAnswerService:  # answers Game of Thrones questions with evidence when there is some
    def __init__(self, generator: GotTextGenerator, retriever: EvidenceRetriever | None, settings: GroundedAnswerSettings,  # wires the model, the optional search, the numbers and the copy guard's memory
                 book_run_index: BookRunIndex | None = None) -> None:  # the copy guard index defaults to none
        self._generator = generator  # the text model
        self._retriever = retriever  # None when retrieval is off or unavailable
        self._settings = settings  # prompt and citation settings
        self._book_run_index = book_run_index  # None when the books are not loaded: the guard then has nothing to compare with

    def _count_question_block_tokens(self, question: str) -> int:  # tokens of the trained question block, the part of the prompt that is never cut
        return count_tokens(render_question_block(question), self._generator.tokenizer)  # the real tokenizer, special words as plain text

    def _require_question_to_fit(self, question: str) -> None:  # raises before any search when the question leaves less than the answer floor of room
        room = self._generator.context_length - self._settings.answer_token_floor - self._settings.safety_margin_tokens  # tokens the question block may use
        if self._count_question_block_tokens(question) > room:  # the block alone is too big
            template_tokens = self._count_question_block_tokens("")  # the fixed header and markers
            raise QuestionTooLongError(max(0, room - template_tokens))  # what is left for the question text itself

    def answer_question(self, answer_request: AnswerRequest, random_generator: torch.Generator | None = None) -> GroundedAnswer:  # searches (or skips), then follows the shared answer path
        self._require_question_to_fit(answer_request.question)  # a question that cannot fit is refused before the search is paid for
        if not answer_request.use_retrieval or self._retriever is None:  # switched off by the request or unavailable
            evidence = build_empty_evidence(EvidenceRoute.RETRIEVAL_OFF)  # nothing searched
        else:  # search the index
            try:  # a search error must never break the answer
                evidence = self._retriever.retrieve(answer_request.question)  # passages and examples
            except Exception as error:  # any failure inside this one retrieval
                logger.error("retrieval failed: %s", type(error).__name__)  # type name only, never text
                evidence = build_empty_evidence(EvidenceRoute.RETRIEVAL_FAILED)  # plain prompt
        return self.answer_from_evidence(answer_request, evidence, self._settings.layout, self._settings.context_token_budget, random_generator)  # shared path

    def answer_from_evidence(self, answer_request: AnswerRequest, evidence: RetrievedEvidence, layout: PromptLayout,  # packs, generates, guards and cites for already retrieved evidence
                             context_token_budget: int, random_generator: torch.Generator | None = None) -> GroundedAnswer:  # optional seeded generator for tests and the evaluation
        settings, timer = self._settings, StageTimer()  # settings and a stage clock
        room_after_question = self._generator.context_length - settings.safety_margin_tokens - self._count_question_block_tokens(answer_request.question)  # tokens left once the question block is in
        answer_reserve = min(max(answer_request.max_new_tokens, settings.answer_token_floor), settings.answer_token_ceiling, room_after_question)  # tokens kept for the answer, never more than the context can hold
        budget = PromptBudget(self._generator.context_length, answer_reserve, context_token_budget, settings.safety_margin_tokens)  # token limits
        with timer.measure("packing"):  # build the prompt
            packed = pack_prompt(answer_request.question, evidence, layout, budget, self._generator.tokenizer)  # prompt within the token budget
        generated = self._generator.generate_answer(  # write the answer
            packed.prompt_token_ids, answer_reserve, answer_request.sampling, answer_request.stop_strings, random_generator,  # prompt, reserve, sampling, stop texts and the seeded generator
        )
        guarded = limit_verbatim_overlap(generated.text, self._book_run_index)  # cut any stretch of the books longer than 25 words, whichever route made the answer
        stages = {**evidence.stage_milliseconds, **timer.stage_milliseconds, "prefill": generated.prefill_milliseconds, "decode": generated.decode_milliseconds}  # retrieval stages plus packing, prefill and decode
        return GroundedAnswer(  # everything the caller needs
            guarded.text, packed.has_context, build_source_citations(packed, settings.book_snippet_enabled),  # answer text, whether evidence reached the prompt, public citations
            self._choose_route(evidence, packed), len(packed.prompt_token_ids), generated.answer_token_count,  # route, prompt length, answer length
            generated.stop_reason, stages, guarded.was_cut, tuple(passage.record.passage_id for passage in packed.passages),  # stop reason, stage times, guard flag, packed passage ids
            tuple(pair.record.pair_id for pair in packed.pair_examples), evidence.best_evidence_score,  # packed pair ids and the gate score
        )

    @staticmethod  # needs no instance
    def _choose_route(evidence: RetrievedEvidence, packed: PackedPrompt) -> EvidenceRoute:  # what the packed prompt holds
        if evidence.route in (EvidenceRoute.RETRIEVAL_OFF, EvidenceRoute.RETRIEVAL_FAILED):  # retrieval never ran
            return evidence.route  # as found
        return choose_evidence_route(bool(packed.passages), bool(packed.pair_examples))  # from the packed evidence
