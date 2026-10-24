# Turns the result dictionaries of the five stages into EVALUATION-REPORT.md (plain tables, no passage text). Missing stages are skipped.
import json  # result files
import math  # confidence margin
from pathlib import Path  # result file paths
from typing import Any, Mapping, Sequence  # inputs

INFLATION_NOTICE = (  # DESIGN.md §2 fact 13: the report must say this
    "G1 answer scores are inflated in BOTH arms (baseline and retrieval): the 2025 starting model of the live weights read the held-out passage "  # first half of the notice text
    "text. Retrieval metrics are not affected."  # second half: the retrieval metrics stay valid
)


def format_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:  # a markdown table as lines
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]  # header and rule
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]  # one line per row
    return lines + [""]  # blank line after the table


def compute_margin_points(rate: float, question_count: int) -> float:  # 95% margin of a rate in percentage points (normal approximation)
    return 196 * math.sqrt(rate * (1 - rate) / question_count) if question_count else 0.0  # 1.96 x standard error x 100


def describe_golden_section(golden: Mapping[str, Any]) -> list[str]:  # how the question sets were cut
    counts = golden["counts"]  # counts written by the golden stage
    rows = [(name, value) for name, value in counts.items()]  # one row per count
    return ["## Question sets", ""] + format_markdown_table(("count", "value"), rows) + [INFLATION_NOTICE, ""]  # section text: heading, table, notice


def describe_retrieval_section(ablation: Mapping[str, Any]) -> list[str]:  # ablation tables and decisions
    rows = [(variant["name"], variant["bundle_name"]) + tuple("%.3f" % variant["metrics"][key] for key in ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10",  # one table row per retrieval variant, metrics rounded to three decimals
            "mean_reciprocal_rank_at_10", "normalized_discounted_cumulative_gain_at_10")) + ("%.0f" % variant["p95_total_milliseconds"],) for variant in ablation["variants"]]  # the last metric names, the Mac p95 time and the loop over the variants
    lines = ["## Retrieval ablation (G1 development)", ""] + format_markdown_table(  # section heading, blank line, then the variant table
        ("variant", "bundle", "hit@1", "hit@3", "hit@5", "hit@10", "reciprocal rank@10", "nDCG@10", "p95 ms (Mac)"), rows)  # variant table
    chunk_rows = [(variant["bundle_name"],) + tuple("%.3f" % variant["metrics"]["hit_within_packed_budget_%d" % budget] for budget in (300, 450, 600)) for variant in ablation["chunk_sizes"]]  # one row per passage size with the hit rate at each budget
    lines += ["Chunk sizes, hit within the packed budget (only measure compared across sizes):", ""] + format_markdown_table(("bundle", "300", "450", "600"), chunk_rows)  # chunk-size table
    decisions = ablation["decisions"]  # the chosen retrieval settings
    lines.append("Chosen: search mode `%s`, reranker %s (%s), chunk size `%s`." % (  # the decisions in one sentence
        decisions["passage_search_mode"], "on" if decisions["reranker_enabled"] else "off", decisions["reranker_reason"], decisions["chunk_size_name"]))  # decision values filled into the sentence above
    test_run = ablation.get("test_run")  # the one run on the reported blocks
    if test_run:  # present once the retrieval stage finished
        hit = test_run["metrics"]["hit_at_5"]  # A5
        lines.append("G1 test (reported, run once): hit@5 %.3f over %d questions, margin about +/- %.1f points (95%%); reciprocal rank@10 %.3f." % (  # G1 test line: hit rate, question count, margin
            hit, test_run["question_count"], compute_margin_points(hit, test_run["question_count"]), test_run["metrics"]["mean_reciprocal_rank_at_10"]))  # filled values: hit, question count, margin, reciprocal rank
    return lines + [""]  # blank line after the section


def describe_calibration_section(calibration: Mapping[str, Any]) -> list[str]:  # floor and pair threshold
    floor, pair = calibration["evidence_floor"], calibration["pair_threshold"]  # the two results
    tradeoff_rows = [("%.2f" % row["target_development_share"], row["floor"], "%.3f" % row["development_share"], "%.3f" % row["fact_development_share"],  # floor trade-off row: coverage target, floor and three shares
                      "%.3f" % row["unanswerable_share"]) for row in calibration.get("floor_tradeoff", [])]  # one row per coverage target
    return ["## Calibration", "",  # section heading and a blank line
            "Evidence floor %.3f (%s score): %.0f%% of G1 development and %.0f%% of fact-development questions keep evidence; %.0f%% of G3 questions still get it." % (  # floor sentence with its shares
                floor["floor"], calibration["gate_score_kind"], 100 * floor["development_share"], 100 * floor["fact_development_share"], 100 * floor["unanswerable_share"]),  # values filled into the floor sentence
            "Pair threshold %s: held-back precision %s, coverage %.2f; on fact-development questions %d examples shown, %d correct." % (  # pair threshold sentence
                pair["threshold"], pair["held_back_precision"], pair["held_back_coverage"], calibration["pair_fact_check"]["shown"], calibration["pair_fact_check"]["correct"]), "",  # values filled into the pair sentence, then a blank line
            "Floor trade-off (A7 needs at most 15% of unanswerable questions with evidence while at least 85% of development questions keep it):", ""  # trade-off table title and a blank line
            ] + format_markdown_table(("development target", "floor", "development", "fact-development", "unanswerable with evidence"), tradeoff_rows)  # table of the trade-off rows follows the title


def describe_answers_section(answers: Mapping[str, Any]) -> list[str]:  # selection grid and the two final runs
    rows = [(score["name"], score["fact_development_majority_right"], "%.3f" % score["held_out_key_term_recall"], "%.0f" % score["mean_prompt_tokens"])  # one row per configuration of the selection grid
            for score in answers["selection"]]  # loop over the selection results
    lines = ["## Answer runs", "", answers["plan"]["reason"], ""] + format_markdown_table(("configuration", "fact-dev right", "G1 dev key-term recall", "prompt tokens"), rows)  # selection table
    final_rows = [(label, arm["facts"]["majority_right"], arm["known_ten"], arm["facts_without_pairs"], "%.3f" % arm["held_out"]["key_term_recall"],  # one row per arm
                   "%.3f" % arm["held_out"]["unsupported_name_rate"], "%.3f" % arm["held_out"]["no_evidence_rate"], arm["copyright_problem_count"])  # last three values of a final-run row
                  for label, arm in (("question_only", answers["final"]["baseline"]), (answers["chosen"], answers["final"]["chosen"]))]  # the two arms shown: the baseline and the chosen configuration
    return lines + ["Final runs (reported once):", ""] + format_markdown_table(  # final comparison table
        ("arm", "facts /30", "known ten /10", "facts /27", "G1 test key-term recall", "invented-name rate", "no-evidence rate", "A16 findings"), final_rows)  # column headings and the rows of the final table


def describe_gate_section(recommended: Mapping[str, Any]) -> list[str]:  # the ship gate and the grounding fine-tune rule
    rows = [(gate_result["gate"], gate_result["status"], gate_result["detail"]) for gate_result in recommended["ship_gate"]]  # one row per gate line
    raft = "needed (%s)" % "; ".join(recommended["raft_reasons"]) if recommended["raft_needed"] else "not needed"  # trigger text
    return ["## Ship gate", ""] + format_markdown_table(("gate", "status", "detail"), rows) + ["Grounding fine-tune: %s." % raft, ""]  # ship gate table plus the grounding fine-tune line


def compose_evaluation_report(results: Mapping[str, Mapping[str, Any] | None]) -> str:  # merges every stage result into one Markdown text
    """results maps 'golden', 'ablation', 'calibration', 'answers', 'recommended' to the stage dictionaries (None when a stage has not run)."""
    sections = (("golden", describe_golden_section), ("ablation", describe_retrieval_section), ("calibration", describe_calibration_section),  # section keys paired with the functions that write them
                ("answers", describe_answers_section), ("recommended", describe_gate_section))  # report order
    lines = ["# Retrieval evaluation report", ""]  # title
    for key, describe in sections:  # every section in order
        if results.get(key):  # only stages that ran
            lines += describe(results[key])  # its lines
    return "\n".join(lines) + "\n"  # the document


def write_json_results(path: Path, payload: Any) -> None:  # one result file: sorted keys, readable, UTF-8
    path.parent.mkdir(parents=True, exist_ok=True)  # the results folder may not exist yet
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")  # stable text


def read_json_results(path: Path) -> Any | None:  # a result file, or None when that stage has not run
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None  # missing = not run
