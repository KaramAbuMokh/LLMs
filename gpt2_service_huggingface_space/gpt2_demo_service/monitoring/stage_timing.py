# Stopwatch for the stages of one answer: wrap a stage in measure() and read the milliseconds per stage name afterwards
import time  # monotonic clock
from contextlib import contextmanager  # lets measure() be used in a with block
from typing import Iterator, Mapping  # type hints


class StageTimer:  # collects how long each named stage took (a stage measured twice adds up)
    def __init__(self) -> None:  # starts with no stages
        self._stage_milliseconds: dict[str, float] = {}  # stage name -> milliseconds so far

    @contextmanager
    def measure(self, stage_name: str) -> Iterator[None]:  # times the with block under this stage name
        started_at = time.perf_counter()  # start of the stage
        try:  # the time counts even when the stage raises
            yield  # the stage runs here
        finally:  # always record
            elapsed_milliseconds = (time.perf_counter() - started_at) * 1000.0  # length of the stage
            self._stage_milliseconds[stage_name] = self._stage_milliseconds.get(stage_name, 0.0) + elapsed_milliseconds  # add to earlier runs

    @property
    def stage_milliseconds(self) -> Mapping[str, float]:  # a copy, so callers cannot change the timer
        return dict(self._stage_milliseconds)  # stage name -> milliseconds
