# Tests of the request gate: line cap, per-visitor window, window expiry, never-negative counter, idle visitor cleanup
from gpt2_demo_service.web_api import request_limits  # module under test (its clock is replaced)
from gpt2_demo_service.web_api.request_limits import PublicLimitSettings, RequestGate  # code under test

LIMITS = PublicLimitSettings(2, 3, 60, 15, 30, 500, 2000)  # two in line, three per visitor per minute


def test_line_cap_refuses_the_next_caller_with_503():  # busy answer
    gate = RequestGate(LIMITS)  # empty line
    assert gate.try_enter("a") is None and gate.try_enter("b") is None  # two places taken
    assert gate.try_enter("c")[0] == 503 and gate.try_enter("c")[2] == 15  # full: status and retry hint
    gate.leave()  # one finishes
    assert gate.try_enter("c") is None  # a place is free again


def test_visitor_rate_limit_and_window_expiry(monkeypatch):  # 429 then allowed again after the window
    now = [1000.0]  # fake clock
    monkeypatch.setattr(request_limits.time, "monotonic", lambda: now[0])  # replace the clock
    gate = RequestGate(PublicLimitSettings(10, 3, 60, 15, 30, 500, 2000))  # roomy line, three per window
    assert [gate.try_enter("a") for _ in range(3)] == [None, None, None]  # three allowed
    assert gate.try_enter("a") == (429, "Too many requests from this address, please slow down.", 30)  # fourth refused
    now[0] += 61.0  # the window passes
    assert gate.try_enter("a") is None  # allowed again


def test_leave_never_goes_below_zero():  # a stray leave is harmless
    gate = RequestGate(LIMITS)  # empty line
    gate.leave()  # nothing to give back
    assert gate.requests_in_line == 0  # still zero


def test_idle_visitors_are_forgotten_when_the_table_grows(monkeypatch):  # memory stays flat
    monkeypatch.setattr(request_limits, "VISITOR_TABLE_LIMIT", 2)  # tiny table
    now = [0.0]  # fake clock
    monkeypatch.setattr(request_limits.time, "monotonic", lambda: now[0])  # replace the clock
    gate = RequestGate(PublicLimitSettings(10, 10, 60, 15, 30, 500, 2000))  # roomy
    for visitor in ("a", "b", "c"):  # three visitors
        gate.try_enter(visitor)  # each asks once
    now[0] = 500.0  # long after the window
    gate.try_enter("d")  # triggers the cleanup
    assert set(gate.recent_request_times) == {"d"}  # the idle three are gone


def test_refused_callers_are_never_stored():  # a 503 or a 429 must not grow the table (memory stays flat under a flood of new addresses)
    gate = RequestGate(PublicLimitSettings(1, 1, 60, 15, 30, 500, 2000))  # one place in line, one request per visitor
    assert gate.try_enter("a") is None  # the only place is taken and "a" is stored
    for flood_address in ("b", "c", "d"):  # strangers arrive while the line is full
        assert gate.try_enter(flood_address)[0] == 503  # refused because the line is full
    assert gate.try_enter("a")[0] == 429  # a known visitor asking again is refused too
    assert set(gate.recent_request_times) == {"a"} and len(gate.recent_request_times["a"]) == 1  # only the admitted visitor, counted once
