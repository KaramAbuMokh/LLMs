# JSON shapes of the API: the request, and two explicit response models (never response_model_exclude_none, which would drop null fields inside sources)
from dataclasses import dataclass  # frozen defaults record
from typing import Annotated  # attaches length limits to list items
from pydantic import BaseModel, Field, StringConstraints  # request validation

PROMPT_SCHEMA_MAX_CHARACTERS = 2000  # hard ceiling of the prompt field; the per-task limits in PublicLimitSettings can only be lower
TASK_NAME_MAX_CHARACTERS = 32  # longest task name accepted: "spam" and "got" are the real ones, anything longer is noise
STOP_TEXT_MAX_CHARACTERS = 40  # longest extra end text


@dataclass(frozen=True)  # immutable record
class RequestDefaults:  # what a request gets for every field it leaves out (declared WITHOUT defaults; the shipped values live in service_settings.py)
    max_new_tokens: int  # answer length in tokens
    temperature: float  # randomness of the text model
    top_p: float  # share of likely words the text model may choose from
    top_k: int  # number of likely words the text model may choose from
    repetition_penalty: float  # discourages the text model from repeating itself


class GenerateRequest(BaseModel):  # the JSON the portfolio sends
    prompt: str = Field(..., min_length=1, max_length=PROMPT_SCHEMA_MAX_CHARACTERS)  # question or email text
    task_name: str = Field(..., max_length=TASK_NAME_MAX_CHARACTERS, description="'spam' classifies an email, anything else answers like Game of Thrones")  # picks the demo
    max_new_tokens: int | None = Field(None, ge=1, le=200, description="answer length requested from the text model (the service clamps it); omitted = the server default")  # answer length
    temperature: float | None = Field(None, ge=0.1, le=2.0, description="randomness of the text model; omitted = the server default")  # randomness
    top_p: float | None = Field(None, ge=0.1, le=1.0, description="share of likely words the text model may choose from; omitted = the server default")  # nucleus size
    top_k: int | None = Field(None, ge=1, le=100, description="number of likely words the text model may choose from; omitted = the server default")  # candidate count
    repetition_penalty: float | None = Field(None, ge=1.0, le=2.0, description="discourages the text model from repeating itself; omitted = the server default")  # repeat penalty
    stop: list[Annotated[str, StringConstraints(min_length=1, max_length=STOP_TEXT_MAX_CHARACTERS)]] = Field(default_factory=list, max_length=4)  # extra end texts (never empty: an empty text would match at once and blank the answer)
    use_retrieval: bool = True  # False asks for the plain answer without evidence from the books


class SourcePayload(BaseModel):  # one source of an answer: all four keys are always present, book/chapter/snippet may be null
    source_type: str  # "book_passage" or "question_answer_pair"
    book: str | None  # book title
    chapter: str | None  # chapter label such as "Tyrion (chapter 12)"
    snippet: str | None  # at most 25 words


class CompletionResponse(BaseModel):  # the spam body: exactly {"completion"}
    completion: str  # the spam label


class GroundedCompletionResponse(BaseModel):  # the Game of Thrones body
    completion: str  # the plain answer (the portfolio reads this key)
    retrieval_used: bool  # True when evidence from the books reached the prompt
    sources: list[SourcePayload]  # what the answer was grounded on


def describe_response_shapes() -> dict:  # documents both bodies of /generate in the OpenAPI page
    return {200: {  # one status code, two possible shapes
        "description": "Spam check: {completion}. Anything else: {completion, retrieval_used, sources}.",  # text shown in the OpenAPI page
        "content": {"application/json": {"schema": {"anyOf": [CompletionResponse.model_json_schema(), GroundedCompletionResponse.model_json_schema()]}}},  # schemas of both possible bodies
    }}
