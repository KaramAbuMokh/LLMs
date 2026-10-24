# Contract check of the deployed Space (DESIGN.md §11.7): the old portfolio body still works, the Game of Thrones body has exactly the new keys,
# the spam body is untouched, errors keep their shape. Run only after a deploy: the Space is public and slow.
import json  # request and response bodies
import urllib.error  # HTTP error statuses carry a body too
import urllib.request  # standard-library client, so the check needs no extra package
from dataclasses import dataclass  # reply record
from typing import Any, Mapping

GOT_REQUEST_BODY = {"prompt": "Who is the mother of dragons?", "max_new_tokens": 50, "task_name": "got"}  # what the live portfolio sends
SPAM_REQUEST_BODY = {"prompt": "Congratulations, you won a free prize, click here now!", "max_new_tokens": 50, "task_name": "spam"}  # spam demo body
TOO_LONG_BODY = {"prompt": "x" * 501, "max_new_tokens": 50, "task_name": "got"}  # over the 500-character question cap: must be a 422
GOT_RESPONSE_KEYS = {"completion", "retrieval_used", "sources"}  # exactly these, never a `text` key
SOURCE_KEYS = {"source_type", "book", "chapter", "snippet"}  # exactly these in every source
OLD_HEALTH_KEYS = {"ok", "models", "weights_repo_commit", "memory_mb", "peak_memory_mb"}  # kept for the existing client
BOOK_SOURCE_TYPE = "book_passage"  # source type that may carry a book snippet


@dataclass(frozen=True)
class HttpReply:  # what the Space answered
    status: int  # HTTP status
    body: Any  # parsed JSON, or None when the body was not JSON
    headers: Mapping[str, str]  # response headers (names lower-cased)


def send_request(url: str, body: Mapping[str, Any] | None, timeout_seconds: float) -> HttpReply:  # GET without a body, POST with one
    data = None if body is None else json.dumps(body).encode("utf-8")  # JSON payload
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")  # the call
    try:  # error statuses are answers too
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # success path
            status, raw_body, headers = response.status, response.read(), response.headers  # reply parts
    except urllib.error.HTTPError as error:  # 4xx and 5xx
        status, raw_body, headers = error.code, error.read(), error.headers  # the same parts
    try:  # the body may not be JSON
        parsed_body = json.loads(raw_body.decode("utf-8"))  # JSON bodies
    except ValueError:  # not JSON
        parsed_body = None  # reported by the checks
    return HttpReply(status, parsed_body, {name.lower(): value for name, value in headers.items()})  # normalised reply


def find_got_body_problems(body: Any) -> list[str]:  # shape rules of a Game of Thrones answer
    if not isinstance(body, dict):  # not an object
        return ["Game of Thrones body is not a JSON object"]  # nothing else can be checked
    problems = []  # findings
    if set(body) != GOT_RESPONSE_KEYS:  # extra or missing keys (a `text` key lands here)
        problems.append("Game of Thrones body keys are %s, expected %s" % (sorted(body), sorted(GOT_RESPONSE_KEYS)))  # report
    if not isinstance(body.get("completion"), str):  # the portfolio reads this
        problems.append("completion is not a string")  # report
    sources = body.get("sources") if isinstance(body.get("sources"), list) else []  # source list
    for number, source in enumerate(sources):  # every source
        if not isinstance(source, dict) or set(source) != SOURCE_KEYS:  # exactly four keys
            problems.append("source %d does not have exactly the keys %s" % (number, sorted(SOURCE_KEYS)))  # report
    if sum(1 for source in sources if isinstance(source, dict) and source.get("source_type") == BOOK_SOURCE_TYPE and source.get("snippet")) > 1:  # copyright rule
        problems.append("more than one book snippet")  # report
    return problems  # empty = fine


def find_health_problems(body: Any) -> list[str]:  # the old health keys must survive
    if not isinstance(body, dict):  # not an object
        return ["health body is not a JSON object"]  # report
    missing = OLD_HEALTH_KEYS - set(body)  # keys the existing client reads
    return ["health lacks the old keys %s" % sorted(missing)] if missing else []  # empty = fine


def check_live_contract(space_url: str, timeout_seconds: float = 120.0) -> list[str]:  # every contract finding of the live Space (empty = fine)
    base_url = space_url.rstrip("/")  # no double slash
    problems = []  # findings
    got_reply = send_request(base_url + "/generate", GOT_REQUEST_BODY, timeout_seconds)  # the portfolio's call
    problems += ["Game of Thrones call returned %d" % got_reply.status] if got_reply.status != 200 else find_got_body_problems(got_reply.body)  # status, then shape
    spam_reply = send_request(base_url + "/generate", SPAM_REQUEST_BODY, timeout_seconds)  # the spam demo
    if spam_reply.status != 200 or spam_reply.body is None or set(spam_reply.body) != {"completion"}:  # body must be exactly {"completion"}
        problems.append("spam call returned %d with body keys %s" % (spam_reply.status, sorted(spam_reply.body) if isinstance(spam_reply.body, dict) else None))  # report
    long_reply = send_request(base_url + "/generate", TOO_LONG_BODY, timeout_seconds)  # an over-long question
    if long_reply.status != 422 or not isinstance(long_reply.body, dict) or "detail" not in long_reply.body:  # unchanged error shape
        problems.append("over-long question returned %d, expected 422 with a detail" % long_reply.status)  # report
    problems += find_health_problems(send_request(base_url + "/health", None, timeout_seconds).body)  # old health keys
    return problems  # 503 is not provoked: it needs a full line, and the Space is public
