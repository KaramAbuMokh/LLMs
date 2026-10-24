# Live checks against a local fake Space (a thread with http.server): a good Space passes, each contract break is reported, the probe reads Server-Timing
import json  # fake bodies
import threading  # server thread
from http.server import BaseHTTPRequestHandler, HTTPServer  # local fake Space
import pytest  # raises
from retrieval_evaluation.live_checks.live_contract_check import check_live_contract, find_got_body_problems, find_health_problems  # under test
from retrieval_evaluation.live_checks.live_latency_probe import parse_server_timing, probe_live_latency  # under test

SOURCE = {"source_type": "book_passage", "book": "B", "chapter": None, "snippet": "a few words"}  # a good source
GOOD_GOT_BODY = {"completion": "Daenerys.", "retrieval_used": True, "sources": [SOURCE]}  # a good Game of Thrones body


def start_fake_space(got_body, spam_body=None):  # serves /generate, /health; returns (server, base url)
    class Handler(BaseHTTPRequestHandler):  # one handler per fake
        def reply(self, status, body, headers=()):  # writes a JSON reply
            raw = json.dumps(body).encode("utf-8")  # payload
            self.send_response(status)  # status line
            self.send_header("Content-Type", "application/json")  # JSON
            for name, value in headers:  # extra headers
                self.send_header(name, value)  # e.g. Server-Timing
            self.end_headers()  # done with headers
            self.wfile.write(raw)  # body

        def do_GET(self):  # /health
            self.reply(200, {"ok": True, "models": ["got"], "weights_repo_commit": "c1", "memory_mb": 1, "peak_memory_mb": 2})  # old keys present

        def do_POST(self):  # /generate
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))  # what was sent
            if len(request["prompt"]) > 500:  # the question cap
                self.reply(422, {"detail": "too long"})  # unchanged error shape
            elif request["task_name"] == "spam":  # spam demo
                self.reply(200, spam_body or {"completion": "spam"})  # exactly the completion
            else:  # Game of Thrones
                self.reply(200, got_body, [("Server-Timing", "retrieval;dur=12.5, prefill;dur=300, decode;dur=4000")])  # with stage times

        def log_message(self, *arguments):  # silence the test output
            pass  # nothing

    server = HTTPServer(("127.0.0.1", 0), Handler)  # random free port
    threading.Thread(target=server.serve_forever, daemon=True).start()  # serve in the background
    return server, "http://127.0.0.1:%d" % server.server_port  # handle and address


def test_a_good_space_has_no_contract_problems():  # old portfolio body, new GoT body, spam body, 422 and health
    server, url = start_fake_space(GOOD_GOT_BODY)  # fake
    try:  # always stop the server
        assert check_live_contract(url, timeout_seconds=5) == []  # nothing to report
    finally:
        server.shutdown()  # stop


def test_each_contract_break_is_reported():  # a `text` key, a loose source, two snippets, a spam body with extra keys
    assert find_got_body_problems(dict(GOOD_GOT_BODY, text="x")) != []  # extra key
    assert find_got_body_problems(dict(GOOD_GOT_BODY, sources=[{"source_type": "book_passage", "book": "B"}])) != []  # source keys
    assert find_got_body_problems(dict(GOOD_GOT_BODY, sources=[SOURCE, SOURCE])) != []  # two book snippets
    assert find_got_body_problems(GOOD_GOT_BODY) == [] and find_health_problems({"ok": True}) != []  # good body, old health keys missing
    server, url = start_fake_space(GOOD_GOT_BODY, {"completion": "spam", "retrieval_used": False})  # spam body with an extra key
    try:  # always stop the server
        assert any("spam" in problem for problem in check_live_contract(url, timeout_seconds=5))  # reported
    finally:
        server.shutdown()  # stop


def test_latency_probe_reads_server_timing_and_refuses_more_than_twenty_calls():  # A9 and A10 inputs
    assert parse_server_timing("retrieval;dur=12.5, prefill;dur=300") == {"retrieval": 12.5, "prefill": 300.0}  # parsed
    server, url = start_fake_space(GOOD_GOT_BODY)  # fake
    try:  # always stop the server
        summary = probe_live_latency(url, ["Who is the mother of dragons?"], call_count=3, timeout_seconds=5)  # three calls
        assert summary.call_count == 3 and summary.failed_call_count == 0 and summary.server_timing_present  # header passed through
        assert summary.stage_p95_milliseconds == {"retrieval": 12.5, "prefill": 300.0, "decode": 4000.0}  # per stage
        with pytest.raises(ValueError):  # the public Space is protected
            probe_live_latency(url, ["q"], call_count=21)  # too many
    finally:
        server.shutdown()  # stop
