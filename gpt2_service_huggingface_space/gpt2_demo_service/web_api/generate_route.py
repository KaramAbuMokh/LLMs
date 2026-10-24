# POST /generate: spam check or Game of Thrones answer, behind the request gate and the single inference lock
import logging  # memory line after each request
from typing import Mapping  # stage timings
from fastapi import APIRouter, HTTPException, Request, Response  # routing
from gpt2_demo_service.grounded_answers.grounded_answer_service import AnswerRequest, GroundedAnswer, QuestionTooLongError  # the answer path and its too-long-question error
from gpt2_demo_service.index_search.evidence_records import RETRIEVAL_STAGE_NAMES  # stage names that count as retrieval
from gpt2_demo_service.language_models.got_text_generator import SamplingSettings  # how words are chosen
from gpt2_demo_service.monitoring.memory_logging import log_memory_usage  # memory after each request
from gpt2_demo_service.monitoring.request_logging import log_answer_event  # one JSON line per answer
from gpt2_demo_service.web_api.api_schemas import CompletionResponse, GenerateRequest, GroundedCompletionResponse, RequestDefaults, SourcePayload, describe_response_shapes  # JSON shapes
from gpt2_demo_service.web_api.visitor_address import find_visitor_address  # the caller's address

router = APIRouter()  # included by the app factory
logger = logging.getLogger(__name__)  # logger for this module


def format_server_timing(stage_milliseconds: Mapping[str, float]) -> str:  # "retrieval;dur=12.3, prefill;dur=45.6, decode;dur=789.0"
    retrieval_milliseconds = sum(stage_milliseconds.get(stage_name, 0.0) for stage_name in RETRIEVAL_STAGE_NAMES)  # all search stages together
    return ", ".join("%s;dur=%.1f" % (name, value) for name, value in (  # the three numbers the latency probe reads
        ("retrieval", retrieval_milliseconds), ("prefill", stage_milliseconds.get("prefill", 0.0)), ("decode", stage_milliseconds.get("decode", 0.0))))  # retrieval, prefill and decode in milliseconds


def fill_omitted(requested: float | int | None, default: float | int) -> float | int:  # the caller's value when it sent one (even a low one), else the server default
    return default if requested is None else requested  # None is the only "not sent"


def build_answer_request(generate_request: GenerateRequest, defaults: RequestDefaults) -> AnswerRequest:  # the request with every omitted field filled from the server defaults
    sampling = SamplingSettings(  # how words are chosen
        fill_omitted(generate_request.temperature, defaults.temperature), fill_omitted(generate_request.top_p, defaults.top_p),  # temperature and nucleus size from the request or the defaults
        fill_omitted(generate_request.top_k, defaults.top_k), fill_omitted(generate_request.repetition_penalty, defaults.repetition_penalty),  # candidate count and repeat penalty from the request or the defaults
    )
    return AnswerRequest(generate_request.prompt, fill_omitted(generate_request.max_new_tokens, defaults.max_new_tokens), sampling, tuple(generate_request.stop), generate_request.use_retrieval)  # the answer path's input


def build_grounded_response(answer: GroundedAnswer) -> GroundedCompletionResponse:  # the Game of Thrones body, every source with all four keys
    sources = [SourcePayload(source_type=str(source.source_type), book=source.book, chapter=source.chapter, snippet=source.snippet) for source in answer.sources]  # citations
    return GroundedCompletionResponse(completion=answer.completion, retrieval_used=answer.retrieval_used, sources=sources)  # body


@router.post("/generate", response_model=None, responses=describe_response_shapes())  # route registration
def generate_completion(generate_request: GenerateRequest, http_request: Request, response: Response) -> CompletionResponse | GroundedCompletionResponse:  # answers one demo request
    state = http_request.app.state  # components, gate, lock and limits set up by the app factory
    is_spam = generate_request.task_name == "spam"  # spam demo, else Game of Thrones
    max_characters = state.limits.email_max_characters if is_spam else state.limits.got_question_max_characters  # limit of this task
    if len(generate_request.prompt) > max_characters:  # text too long
        raise HTTPException(status_code=422, detail="%s can be up to %d characters." % ("Emails" if is_spam else "Questions", max_characters))  # text over this task's limit
    refusal = state.request_gate.try_enter(find_visitor_address(http_request))  # ask for a place in the line
    if refusal:  # the line is full or this visitor is too fast
        status_code, message, retry_seconds = refusal  # unpack the refusal
        raise HTTPException(status_code=status_code, detail=message, headers={"Retry-After": str(retry_seconds)})  # tell the caller when to come back
    try:  # always give the place back
        with state.inference_lock:  # wait for any running generation to finish
            if is_spam:  # spam demo
                result: CompletionResponse | GroundedCompletionResponse = CompletionResponse(completion=state.components.spam_classifier.label_email(generate_request.prompt))  # spam label in the spam body
            else:  # Game of Thrones demo
                answer = state.components.answer_service.answer_question(build_answer_request(generate_request, state.request_defaults))  # Game of Thrones answer with evidence when found
                log_answer_event(answer, generate_request.prompt, state.components.index_version)  # one JSON line, no visitor text
                response.headers["Server-Timing"] = format_server_timing(answer.stage_milliseconds)  # lets the latency probe read stage times
                result = build_grounded_response(answer)  # body with sources
    except QuestionTooLongError as error:  # the question block cannot fit the model's context
        raise HTTPException(status_code=422, detail="Questions can be up to %d tokens." % error.max_question_tokens) from error  # same shape as the character limit
    finally:  # runs on success and on errors
        state.request_gate.leave()  # free the place in the line
    log_memory_usage("after_spam" if is_spam else "after_got")  # memory after the request, labelled by a fixed word (visitor text never reaches the log)
    return result  # JSON answer
