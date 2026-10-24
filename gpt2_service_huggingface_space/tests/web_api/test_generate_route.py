# Tests of POST /generate: the portfolio contract, exact key sets, error shapes, Server-Timing, OpenAPI, limits
import logging  # log level control
import re  # header format
from dataclasses import replace  # copies settings with changes

GOT_BODY = {"prompt": "Where are the keys of the salt tower kept?", "task_name": "got", "max_new_tokens": 50}  # the old portfolio body
SOURCE_KEYS = {"source_type", "book", "chapter", "snippet"}  # every source has exactly these


def test_old_portfolio_body_still_works(make_client):  # 200, string completion, no "text" key
    body = make_client().post("/generate", json=GOT_BODY).json()  # ask
    assert isinstance(body["completion"], str) and "text" not in body  # the portfolio reads completion


def test_got_body_has_exactly_three_keys_and_each_source_four(make_client):  # D17
    body = make_client().post("/generate", json=GOT_BODY).json()  # ask
    assert set(body) == {"completion", "retrieval_used", "sources"} and body["retrieval_used"] is True  # evidence was found
    assert body["sources"] and all(set(source) == SOURCE_KEYS for source in body["sources"])  # four keys each
    assert sum(1 for source in body["sources"] if source["source_type"] == "book_passage" and source["snippet"]) == 1  # one book snippet


def test_source_without_book_or_chapter_still_has_all_four_keys(make_client):  # null values stay present
    body = make_client().post("/generate", json={"prompt": "Who rules the Glass Coast?", "task_name": "got"}).json()  # matches the unlocated pair
    unlocated = [source for source in body["sources"] if source["book"] is None]  # the pair with no book
    assert unlocated and all(set(source) == SOURCE_KEYS and source["chapter"] is None for source in unlocated)  # keys present, values null


def test_spam_body_is_exactly_completion(make_client):  # A12: unchanged spam body
    client = make_client()  # app
    assert client.post("/generate", json={"prompt": "You are a WINNER, click now", "task_name": "spam"}).json() == {"completion": "spam"}  # exact body
    assert client.post("/generate", json={"prompt": "See you at lunch", "task_name": "spam"}).json() == {"completion": "not spam"}  # exact body


def test_use_retrieval_false_gives_the_plain_answer(make_client):  # opt-out per request
    body = make_client().post("/generate", json={**GOT_BODY, "use_retrieval": False}).json()  # ask
    assert body["retrieval_used"] is False and body["sources"] == []  # plain prompt, empty list (not missing)


def test_server_timing_header_on_got_answers_only(make_client):  # the latency probe reads it
    client = make_client()  # app
    header = client.post("/generate", json=GOT_BODY).headers.get("server-timing", "")  # header of a GoT answer
    assert re.fullmatch(r"retrieval;dur=[\d.]+, prefill;dur=[\d.]+, decode;dur=[\d.]+", header)  # three numbers
    assert "server-timing" not in client.post("/generate", json={"prompt": "hello", "task_name": "spam"}).headers  # spam path untouched


def test_error_shapes_are_unchanged(make_client):  # 422 for long text and bad fields
    client = make_client()  # app
    long_question = client.post("/generate", json={**GOT_BODY, "prompt": "x" * 501})  # over 500 characters
    assert long_question.status_code == 422 and long_question.json() == {"detail": "Questions can be up to 500 characters."}  # same message
    assert client.post("/generate", json={"prompt": "hi"}).status_code == 422  # missing task_name
    assert client.post("/generate", json={"prompt": "x" * 2001, "task_name": "spam"}).status_code == 422  # over the schema ceiling


def test_lowered_limits_are_enforced(make_client, test_settings):  # the settings can only lower the schema ceiling
    settings = replace(test_settings, limits=replace(test_settings.limits, got_question_max_characters=20, email_max_characters=30))  # tight limits
    client = make_client(settings)  # app
    assert client.post("/generate", json=GOT_BODY).status_code == 422  # question over 20
    assert client.post("/generate", json={"prompt": "x" * 31, "task_name": "spam"}).json() == {"detail": "Emails can be up to 30 characters."}  # email over 30


def test_full_line_gives_503_with_retry_after(make_client):  # the line cap
    client = make_client()  # app
    client.app.state.request_gate.requests_in_line = client.app.state.limits.max_requests_in_line  # the line is full
    response = client.post("/generate", json=GOT_BODY)  # next caller
    assert response.status_code == 503 and response.headers["retry-after"] == "15"  # busy


def test_fast_visitor_gets_429_with_retry_after(make_client, test_settings):  # per-visitor rate limit
    client = make_client(replace(test_settings, limits=replace(test_settings.limits, max_requests_per_visitor=2)))  # two per window
    headers = {"x-forwarded-for": "203.0.113.9"}  # one visitor
    statuses = [client.post("/generate", json=GOT_BODY, headers=headers).status_code for _ in range(3)]  # three calls
    assert statuses == [200, 200, 429]  # third refused
    assert client.post("/generate", json=GOT_BODY, headers=headers).headers["retry-after"] == "30"  # wait hint


def test_place_in_line_is_given_back_after_a_model_error(make_client, spy_generator):  # a 500 must not leak a place
    spy_generator.generate_answer = lambda *arguments, **keywords: (_ for _ in ()).throw(RuntimeError("model failed"))  # generator raises
    client = make_client()  # app
    assert client.post("/generate", json=GOT_BODY).status_code == 500  # model errors still return 500
    assert client.app.state.request_gate.requests_in_line == 0  # place returned


def test_openapi_documents_both_response_shapes(make_client):  # response_model=None, shapes in responses=
    schema = make_client().get("/openapi.json").json()["paths"]["/generate"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]  # documented body
    property_sets = [set(option["properties"]) for option in schema["anyOf"]]  # keys of each shape
    assert {"completion"} in property_sets and {"completion", "retrieval_used", "sources"} in property_sets  # both documented


def test_spoofed_forwarded_for_header_cannot_reset_the_visitor_limit(make_client, test_settings):  # the live probe of 2026-10-25: a new fake first entry gave a fresh allowance
    client = make_client(replace(test_settings, limits=replace(test_settings.limits, max_requests_per_visitor=2)))  # two per window
    statuses = [client.post("/generate", json=GOT_BODY, headers={"x-forwarded-for": "198.51.100.%d, 203.0.113.9" % number}).status_code for number in range(1, 4)]  # a new spoofed first entry each time
    assert statuses == [200, 200, 429]  # the proxy-written last entry counts, so the third call is refused


def test_a_question_too_long_for_the_model_gets_422_and_gives_the_place_back(make_client):  # the live probe: 500 emoji were 1,533 tokens and gave a 500
    client = make_client()  # app (the spy generator reports a 1,024 token context)
    response = client.post("/generate", json={**GOT_BODY, "prompt": "\U0001F409" * 500})  # 500 characters, so it passes the character limit
    assert response.status_code == 422 and response.json()["detail"].startswith("Questions can be up to ") and response.json()["detail"].endswith(" tokens.")  # same shape as the character limit
    assert client.app.state.request_gate.requests_in_line == 0  # the place in line is back


def test_empty_stop_text_and_long_task_name_are_refused(make_client):  # fields that used to cause empty answers and log noise
    client = make_client()  # app
    assert client.post("/generate", json={**GOT_BODY, "stop": [""]}).status_code == 422  # an empty stop text would end the answer at once
    assert client.post("/generate", json={**GOT_BODY, "task_name": "got" * 20}).status_code == 422  # task names are at most 32 characters
    assert client.post("/generate", json={**GOT_BODY, "stop": ["Question:"]}).status_code == 200  # a real stop text still works


def test_omitted_fields_use_the_server_defaults_and_explicit_ones_win(make_client, spy_generator, test_settings):  # temperature default lives in service_settings.py
    seen = []  # sampling settings the model was given
    original = spy_generator.generate_answer  # the scripted generator
    spy_generator.generate_answer = lambda ids, tokens, sampling, stops, rng=None: (seen.append(sampling), original(ids, tokens, sampling, stops, rng))[1]  # records the sampling
    client = make_client()  # app
    client.post("/generate", json={"prompt": "Where are the keys of the salt tower kept?", "task_name": "got"})  # the portfolio sends no sampling fields
    client.post("/generate", json={**GOT_BODY, "temperature": 0.1})  # an explicit low value must survive
    defaults = test_settings.request_defaults  # server values
    assert (seen[0].temperature, seen[0].top_p, seen[0].top_k, seen[0].repetition_penalty) == (defaults.temperature, defaults.top_p, defaults.top_k, defaults.repetition_penalty)  # all four from the server
    assert seen[1].temperature == 0.1 and seen[1].top_p == defaults.top_p  # the request's own temperature wins


def test_the_log_after_a_request_never_holds_caller_text(make_client, caplog):  # README: "Logs hold no visitor text"
    caplog.set_level(logging.INFO)  # capture info lines
    client = make_client()  # app
    client.post("/generate", json={**GOT_BODY, "task_name": "got"})  # a Game of Thrones call
    client.post("/generate", json={"prompt": "hello", "task_name": "spam" + "\nfake log line"})  # a task name with a newline is a Game of Thrones call (only "spam" is spam)
    client.post("/generate", json={"prompt": "hello", "task_name": "spam"})  # a real spam call
    memory_lines = [record.getMessage() for record in caplog.records if "memory at after_" in record.getMessage()]  # one line per answered request
    assert all(line.startswith(("memory at after_got", "memory at after_spam")) for line in memory_lines) and memory_lines  # fixed labels only
    assert not any("fake log line" in record.getMessage() for record in caplog.records)  # the injected newline never reached the log
