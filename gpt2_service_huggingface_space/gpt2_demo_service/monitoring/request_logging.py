# One JSON log line per Game of Thrones answer: ids, scores, route, timings and token counts; never the question or any book text
import hashlib  # short question fingerprint
import json  # the log line
import logging  # writes to the Space log (which must be treated as public)
from typing import TYPE_CHECKING  # import for type checkers only

if TYPE_CHECKING:  # avoids a runtime dependency on the answer package
    from gpt2_demo_service.grounded_answers.grounded_answer_service import GroundedAnswer  # the finished answer

logger = logging.getLogger(__name__)  # logger for this module
QUESTION_HASH_LENGTH = 8  # hex characters of the question fingerprint


def build_answer_event(answer: "GroundedAnswer", question: str, index_version: str | None) -> dict:  # the fields of one log line
    return {
        "event": "got_answer",  # lets a log reader filter these lines
        "question_characters": len(question),  # length only
        "question_hash": hashlib.sha256(question.encode("utf-8")).hexdigest()[:QUESTION_HASH_LENGTH],  # lets repeats be counted, not read
        "route": str(answer.route),  # what retrieval found and what the prompt held
        "retrieval_used": answer.retrieval_used,  # evidence reached the prompt
        "evidence_score": answer.best_evidence_score,  # the gate score
        "passage_ids": list(answer.passage_ids),  # ids of the packed passages
        "pair_ids": list(answer.pair_ids),  # ids of the packed examples
        "stage_milliseconds": {name: round(value, 1) for name, value in answer.stage_milliseconds.items()},  # time per stage
        "prompt_token_count": answer.prompt_token_count,  # tokens in the prompt
        "answer_token_count": answer.answer_token_count,  # tokens generated
        "stop_reason": answer.stop_reason,  # why generation ended
        "verbatim_guard_cut": answer.verbatim_guard_cut,  # the copy guard shortened the answer
        "index_version": index_version,  # which bundle was searched (None when retrieval is off)
    }


def log_answer_event(answer: "GroundedAnswer", question: str, index_version: str | None) -> None:  # writes the line
    logger.info(json.dumps(build_answer_event(answer, question, index_version), sort_keys=True))  # one line of JSON
