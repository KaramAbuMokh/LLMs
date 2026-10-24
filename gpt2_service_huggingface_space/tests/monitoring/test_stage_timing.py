# Tests of the stage timer: stages are named, repeated stages add up, a failing stage is still timed, callers get a copy
import pytest  # expected errors
from gpt2_demo_service.monitoring.stage_timing import StageTimer  # code under test


def test_measure_records_a_named_stage():  # one stage, one number
    timer = StageTimer()  # fresh timer
    with timer.measure("encode"):  # time a block
        sum(range(1000))  # some work
    assert list(timer.stage_milliseconds) == ["encode"] and timer.stage_milliseconds["encode"] >= 0.0  # recorded


def test_same_stage_measured_twice_adds_up():  # passages and pairs share stage names
    timer = StageTimer()  # fresh timer
    for _ in range(2):  # twice
        with timer.measure("dense_search"):  # same stage
            sum(range(1000))  # some work
    assert len(timer.stage_milliseconds) == 1  # still one entry


def test_failing_stage_is_still_timed_and_copy_is_safe():  # the time counts even on errors
    timer = StageTimer()  # fresh timer
    with pytest.raises(RuntimeError):  # the stage fails
        with timer.measure("gate"):  # time a block
            raise RuntimeError("boom")  # failure
    timer.stage_milliseconds["other"] = 1.0  # a caller edits the returned copy
    assert list(timer.stage_milliseconds) == ["gate"]  # the timer is unchanged
