# Tests of the per-answer log line: spelled-out keys, no question text, no passage text
import json  # parse the line
import logging  # capture log records
from gpt2_demo_service.grounded_answers.grounded_answer_service import GroundedAnswer  # answer record
from gpt2_demo_service.grounded_answers.source_citations import SourceCitation  # citation record
from gpt2_demo_service.index_search.evidence_records import EvidenceRoute  # route names
from gpt2_demo_service.monitoring.request_logging import log_answer_event  # code under test
from retrieval_contracts.bundle_records import SourceType  # source kinds

EXPECTED_KEYS = {  # DESIGN.md 7.8, every key spelled out
    "event", "question_characters", "question_hash", "route", "retrieval_used", "evidence_score", "passage_ids", "pair_ids",
    "stage_milliseconds", "prompt_token_count", "answer_token_count", "stop_reason", "verbatim_guard_cut", "index_version",
}


def test_log_line_has_exactly_the_documented_keys_and_no_text(caplog):  # one JSON line per answer
    answer = GroundedAnswer(  # an answer whose texts must not reach the log
        "SECRET ANSWER TEXT", True, (SourceCitation(SourceType.BOOK_PASSAGE, "Book", "Chapter (chapter 1)", "SECRET SNIPPET"),),
        EvidenceRoute.PASSAGES_ONLY, 321, 45, "end_token", {"encode": 1.234, "decode": 99.96}, False, ("bp-0-001-000",), (), 0.71,
    )
    with caplog.at_level(logging.INFO, logger="gpt2_demo_service.monitoring.request_logging"):  # capture
        log_answer_event(answer, "SECRET QUESTION TEXT", "got-retrieval-20261024-eeb6e60a")  # log
    line = caplog.records[0].getMessage()  # the one line
    event = json.loads(line)  # valid JSON
    assert set(event) == EXPECTED_KEYS and event["question_characters"] == 20 and len(event["question_hash"]) == 8  # keys and fingerprint
    assert event["stage_milliseconds"] == {"encode": 1.2, "decode": 100.0} and event["route"] == "passages_only"  # rounded stages, route name
    assert "SECRET" not in line  # no question, answer or snippet text
