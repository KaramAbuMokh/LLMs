# Keeps the public Space safe on 2 CPU cores: caps how many requests wait in line and how often one visitor may ask
import threading  # lock around the shared counters
import time  # clock for the per-visitor window
from collections import deque  # per-visitor request times
from dataclasses import dataclass  # frozen settings record

VISITOR_TABLE_LIMIT = 5000  # past this many addresses, old ones are forgotten so memory stays flat


@dataclass(frozen=True)  # immutable record
class PublicLimitSettings:  # the public-safety numbers (shipped values live in service_settings.py)
    max_requests_in_line: int  # requests running or waiting; the next one gets 503
    max_requests_per_visitor: int  # most requests one visitor address may send in the window
    rate_window_seconds: int  # length of the per-visitor window
    busy_retry_seconds: int  # what a busy Space tells the caller to wait
    rate_retry_seconds: int  # what a too-fast visitor is told to wait
    got_question_max_characters: int  # longest Game of Thrones question accepted
    email_max_characters: int  # longest email accepted by the spam check


class RequestGate:  # decides whether a new request may join the line
    def __init__(self, limits: PublicLimitSettings) -> None:  # starts with an empty line
        self.limits = limits  # the numbers to enforce
        self.counter_lock = threading.Lock()  # protects the counter and the table below
        self.requests_in_line = 0  # requests running or waiting right now
        self.recent_request_times: dict[str, deque[float]] = {}  # visitor address -> times of its recent ADMITTED requests (refused callers are never stored)

    def try_enter(self, visitor_address: str) -> tuple[int, str, int] | None:  # None when allowed, else (status_code, message, retry_seconds)
        now = time.monotonic()  # current time for the window
        with self.counter_lock:  # one decision at a time
            request_times = self.recent_request_times.get(visitor_address, deque())  # this visitor's recent requests; a stranger gets a throwaway deque, nothing is stored yet
            while request_times and now - request_times[0] > self.limits.rate_window_seconds:  # drop requests older than the window
                request_times.popleft()  # forget the oldest one
            if len(request_times) >= self.limits.max_requests_per_visitor:  # this visitor is asking too fast
                return 429, "Too many requests from this address, please slow down.", self.limits.rate_retry_seconds  # slow down
            if self.requests_in_line >= self.limits.max_requests_in_line:  # the line is full
                return 503, "The model is busy answering other visitors, please try again in a few seconds.", self.limits.busy_retry_seconds  # come back later
            self.recent_request_times.setdefault(visitor_address, request_times).append(now)  # admitted: store the visitor and count this request
            self.requests_in_line += 1  # take a place in the line
            self.forget_idle_visitors(now)  # keep the table small
            return None  # allowed

    def leave(self) -> None:  # gives the place in the line back once the answer is sent
        with self.counter_lock:  # one change at a time
            self.requests_in_line = max(0, self.requests_in_line - 1)  # never below zero

    def forget_idle_visitors(self, now: float) -> None:  # removes visitors with no request inside the window (call with the lock held)
        if len(self.recent_request_times) <= VISITOR_TABLE_LIMIT:  # table is still small
            return  # nothing to clean
        idle_addresses = [  # addresses whose newest request is outside the window
            address for address, times in self.recent_request_times.items()  # every address and its request times
            if not times or now - times[-1] > self.limits.rate_window_seconds  # idle: no request left, or the newest one is outside the window
        ]
        for address in idle_addresses:  # drop each idle address
            del self.recent_request_times[address]  # free its memory
