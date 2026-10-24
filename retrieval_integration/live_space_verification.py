# Checks the deployed Space with NO token, as a visitor would: health, the portfolio contract, the spam path, the 30 fact questions with timing, memory
# Run from the project folder: PYTHONPATH=gpt2_service_huggingface_space:. python retrieval_integration/live_space_verification.py [--space-url URL] [--output-name NAME]
import argparse  # command line
import json  # result file
import time  # call timing and pacing
from pathlib import Path  # result location
from retrieval_evaluation.evaluation_settings import EvaluationSettings  # paths of the question sets
from retrieval_evaluation.evaluation_stages.answers_stage import build_lookup_helpers  # the independent copyright check against ALL passages (A16)
from retrieval_evaluation.evaluation_stages.golden_sets_stage import load_golden_sets, load_rule_book  # the 30 fact questions and their word rules
from retrieval_evaluation.live_checks.live_contract_check import check_live_contract, find_got_body_problems, send_request  # contract helpers
from retrieval_evaluation.live_checks.live_latency_probe import parse_server_timing  # Server-Timing header reader
from retrieval_evaluation.metrics.ranking_metrics import percentile_value  # nearest-rank percentile

SPACE_URL = "https://karamam-gpt2-demos.hf.space"  # the public Space
RESULTS_FOLDER = Path(__file__).resolve().parent / "results"  # private result files
MINIMUM_SECONDS_BETWEEN_CALLS = 3.3  # keeps one caller under the 20-calls-per-minute limit
SPAM_SAMPLE_EMAILS = ("Congratulations, you won a free prize, click here now!", "Hi Sam, the meeting moved to 3 pm tomorrow, see you there.")  # one spam-like, one normal
SPAM_CALLS = 6  # calls for the spam p95 (A12)
SERVED_BUNDLE_NAME = "target_128"  # the local copy of the bundle the Space serves (for the A16 check)


def timed_call(url: str, body: dict | None) -> tuple[object, float]:  # (reply, seconds) of one HTTP call
    started = time.perf_counter()  # call clock
    reply = send_request(url, body, 120.0)  # the call
    return reply, time.perf_counter() - started  # reply and seconds


def verify_fact_questions(base_url: str, settings: EvaluationSettings) -> dict:  # the 30 questions live, judged by the same word rules as the Kaggle scoring
    golden, rule_book = load_golden_sets(settings), load_rule_book(settings)  # questions and judge
    _, find_copyright_problems = build_lookup_helpers(dict(settings.locations.bundle_folders)[SERVED_BUNDLE_NAME])  # A16 check over every passage of the served bundle
    records, last_start = [], 0.0  # collected answers and pacing clock
    for question in golden.fact:  # one call per question
        time.sleep(max(0.0, MINIMUM_SECONDS_BETWEEN_CALLS - (time.monotonic() - last_start)))  # stay under the visitor rate limit
        last_start = time.monotonic()  # this call starts now
        reply, seconds = timed_call(base_url + "/generate", {"prompt": question.rule.question, "task_name": "got", "max_new_tokens": 120})  # the portfolio's call shape
        body = reply.body if isinstance(reply.body, dict) else {}  # empty on an error
        completion = body.get("completion", "") if isinstance(body.get("completion"), str) else ""  # the answer text
        records.append({"question_id": question.question_id, "status": reply.status, "seconds": round(seconds, 2), "is_right": rule_book.is_answer_right(question.rule, completion) if reply.status == 200 else False,  # judged
                        "is_known_ten": question.is_known_ten, "is_word_for_word_in_pairs": question.is_word_for_word_in_pairs, "retrieval_used": body.get("retrieval_used"),  # flags
                        "source_count": len(body.get("sources", [])), "copyright_problems": find_copyright_problems(completion, body.get("sources", [])), "shape_problems": find_got_body_problems(body) if reply.status == 200 else ["status %d" % reply.status],  # contract
                        "server_timing_milliseconds": parse_server_timing(reply.headers.get("server-timing", "")), "completion": completion, "sources": body.get("sources", [])})  # evidence
    seconds_list = [record["seconds"] for record in records if record["status"] == 200]  # successful calls
    return {"records": records, "right_count": sum(record["is_right"] for record in records), "known_ten_right": sum(record["is_right"] and record["is_known_ten"] for record in records),  # scores
            "right_without_pair_questions": sum(record["is_right"] and not record["is_word_for_word_in_pairs"] for record in records), "failed_call_count": len(records) - len(seconds_list), "copyright_problem_count": sum(len(record["copyright_problems"]) for record in records),  # A1-A3 and A16
            "p50_seconds": percentile_value(seconds_list, 50.0), "p95_seconds": percentile_value(seconds_list, 95.0), "retrieval_used_count": sum(bool(record["retrieval_used"]) for record in records),  # A10
            "stage_p95_milliseconds": {stage: percentile_value([record["server_timing_milliseconds"][stage] for record in records if stage in record["server_timing_milliseconds"]], 95.0)
                                       for stage in {name for record in records for name in record["server_timing_milliseconds"]}}}  # A9


def verify_spam_path(base_url: str) -> dict:  # spam answers, exact body shape and p95 (A12)
    seconds_list, labels, shapes_exact = [], [], True  # collected
    for call_number in range(SPAM_CALLS):  # a handful of calls
        time.sleep(MINIMUM_SECONDS_BETWEEN_CALLS)  # pacing
        reply, seconds = timed_call(base_url + "/generate", {"prompt": SPAM_SAMPLE_EMAILS[call_number % 2], "task_name": "spam", "max_new_tokens": 50})  # spam call
        seconds_list.append(seconds)  # timing
        shapes_exact = shapes_exact and reply.status == 200 and isinstance(reply.body, dict) and set(reply.body) == {"completion"}  # body exactly {"completion"}
        labels.append(reply.body.get("completion") if isinstance(reply.body, dict) else None)  # what it said
    return {"labels": labels, "body_exactly_completion": shapes_exact, "p95_seconds": percentile_value(seconds_list, 95.0)}  # summary


def verify_space(base_url: str) -> dict:  # everything, in the order a visitor would hit it
    settings = EvaluationSettings()  # paths
    health_before = send_request(base_url + "/health", None, 120.0).body  # state before the fact run
    contract_problems = check_live_contract(base_url)  # portfolio body, spam body, 422 shape, old health keys
    facts = verify_fact_questions(base_url, settings)  # 30 live questions
    spam = verify_spam_path(base_url)  # spam path
    health_after = send_request(base_url + "/health", None, 120.0).body  # memory after the run
    return {"health_before": health_before, "contract_problems": contract_problems, "facts": facts, "spam": spam, "health_after": health_after}  # whole record


if __name__ == "__main__":  # run as a script
    parser = argparse.ArgumentParser(description="Token-free live verification of the GPT-2 demos Space")  # parser
    parser.add_argument("--space-url", default=SPACE_URL, help="Space base URL")  # target
    parser.add_argument("--output-name", default="live_space_verification.json", help="result file name inside results/")  # output
    options = parser.parse_args()  # parsed
    record = verify_space(options.space_url.rstrip("/"))  # run
    (RESULTS_FOLDER / options.output_name).write_text(json.dumps(record, indent=1), encoding="utf-8")  # keep the full record, private
    facts = record["facts"]  # shortcut
    print(json.dumps({"retrieval_health": record["health_before"].get("retrieval"), "memory_mb": [record["health_after"].get("memory_mb"), record["health_after"].get("peak_memory_mb")],  # summary
                      "contract_problems": record["contract_problems"], "facts_right": facts["right_count"], "known_ten_right": facts["known_ten_right"], "right_without_pair_questions": facts["right_without_pair_questions"],
                      "failed_calls": facts["failed_call_count"], "copyright_problems": facts["copyright_problem_count"], "p50_p95_seconds": [facts["p50_seconds"], facts["p95_seconds"]], "stage_p95_milliseconds": facts["stage_p95_milliseconds"],
                      "retrieval_used_count": facts["retrieval_used_count"], "spam": record["spam"]}, indent=1))
