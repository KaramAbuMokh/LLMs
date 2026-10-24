# Latency probe of the deployed Space (DESIGN.md §11.7): one caller, at most 20 calls, total time and the Server-Timing stages.
import re  # Server-Timing parsing
import time  # call timing
from dataclasses import dataclass  # summary record
from typing import Mapping, Sequence
from retrieval_evaluation.live_checks.live_contract_check import send_request  # the shared HTTP helper
from retrieval_evaluation.metrics.ranking_metrics import percentile_value  # nearest-rank percentile

MAX_PROBE_CALLS = 20  # the Space is public and slow: never more than this
STAGE_PATTERN = re.compile(r"([A-Za-z_]+);dur=([0-9.]+)")  # "retrieval;dur=12.5"


@dataclass(frozen=True)
class LatencySummary:  # what the probe measured
    call_count: int  # calls made
    failed_call_count: int  # calls that did not return 200
    total_p50_seconds: float  # median seconds per call
    total_p95_seconds: float  # 95th percentile seconds per call (A10)
    stage_p95_milliseconds: Mapping[str, float]  # per Server-Timing stage
    server_timing_present: bool  # the edge passed the header through


def parse_server_timing(header_value: str) -> dict[str, float]:  # stage name -> milliseconds
    return {name: float(duration) for name, duration in STAGE_PATTERN.findall(header_value)}  # every "name;dur=value"


def probe_live_latency(space_url: str, questions: Sequence[str], call_count: int = MAX_PROBE_CALLS, timeout_seconds: float = 120.0) -> LatencySummary:  # one caller, at most 20 calls
    if call_count > MAX_PROBE_CALLS or call_count < 1:  # protect the public Space
        raise ValueError("call_count must be between 1 and %d" % MAX_PROBE_CALLS)  # refuse
    totals, stage_values, failed = [], {}, 0  # measurements so far
    for call_number in range(call_count):  # one caller, one call at a time
        body = {"prompt": questions[call_number % len(questions)], "max_new_tokens": 120, "task_name": "got"}  # cycle the questions
        started = time.perf_counter()  # call clock
        reply = send_request(space_url.rstrip("/") + "/generate", body, timeout_seconds)  # the call
        totals.append(time.perf_counter() - started)  # seconds taken
        failed += reply.status != 200  # count failures
        for stage, milliseconds in parse_server_timing(reply.headers.get("server-timing", "")).items():  # header stages, if any
            stage_values.setdefault(stage, []).append(milliseconds)  # collect
    return LatencySummary(  # the summary
        call_count, failed, percentile_value(totals, 50.0), percentile_value(totals, 95.0),
        {stage: percentile_value(values, 95.0) for stage, values in stage_values.items()}, bool(stage_values),
    )
