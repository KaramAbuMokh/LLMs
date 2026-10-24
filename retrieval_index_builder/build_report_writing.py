# Collects the facts of one build (alignment, passages, pair placement, chapter rates, encoder parity, timings) into build_report.json
import json  # report file
from collections import Counter  # per-book tallies
from pathlib import Path  # report and parity files
from typing import Mapping, Sequence  # inputs
from retrieval_contracts.bundle_manifest import BundleManifest  # counts, sizes and version
from retrieval_contracts.bundle_records import PassageRecord, QuestionAnswerRecord  # records to summarise

PERCENTILES = (5, 25, 50, 75, 95)  # token percentiles shown for the passages


def describe_passage_tokens(passages: Sequence[PassageRecord]) -> dict[str, object]:  # count, token percentiles and the share without a chapter
    token_counts = sorted(passage.token_count for passage in passages)  # ascending sizes
    percentiles = {"p%d" % percent: token_counts[min(len(token_counts) - 1, len(token_counts) * percent // 100)] for percent in PERCENTILES}  # nearest rank
    return {  # what the report shows
        "passage_count": len(passages), "minimum_tokens": token_counts[0], "maximum_tokens": token_counts[-1], "total_tokens": sum(token_counts), **percentiles,  # counts and sizes of the passages
        "passages_without_chapter": sum(1 for passage in passages if passage.chapter_number is None),  # appendix text and unaligned books
    }


def describe_pair_chapter_rates(pairs: Sequence[QuestionAnswerRecord], placement_counts: Mapping[str, int]) -> dict[str, object]:  # chapter rate overall and per book
    pairs_by_book, pairs_with_chapter_by_book = Counter(), Counter()  # tallies keyed by book title
    for pair in pairs:  # every pair
        book_title = pair.book_title or "unplaced"  # unplaced pairs have no book
        pairs_by_book[book_title] += 1  # pairs of that book
        pairs_with_chapter_by_book[book_title] += 1 if pair.chapter_number is not None else 0  # of which carry a chapter
    return {  # what the report shows
        "placement_counts_by_method": dict(placement_counts), "pair_count": len(pairs),  # how the pairs were placed
        "pairs_with_chapter": sum(pairs_with_chapter_by_book.values()), "chapter_rate": round(sum(pairs_with_chapter_by_book.values()) / max(1, len(pairs)), 4),  # how many pairs carry a chapter and the share
        "chapter_rate_by_book": {book: round(pairs_with_chapter_by_book[book] / count, 4) for book, count in sorted(pairs_by_book.items())},  # the share per book
        "pairs_by_book": dict(sorted(pairs_by_book.items())),  # pairs per book
    }


def write_build_report(report_path: Path, manifest: BundleManifest, alignment_report: Mapping[str, object], passages: Sequence[PassageRecord],  # report writer: the facts of one build go into one JSON file
                       pairs: Sequence[QuestionAnswerRecord], placement_counts: Mapping[str, int], stage_seconds: Mapping[str, float], parity_path: Path) -> None:  # inputs: records, counts, timings and the parity result file
    report = {  # one JSON document, no book text
        "index_version": manifest.index_version, "built_at_utc": manifest.built_at_utc,  # which bundle this report describes
        "chunking": {"target_tokens": manifest.chunking.target_tokens, "max_tokens": manifest.chunking.max_tokens, "overlap_sentences": manifest.chunking.overlap_sentences},  # chunk sizes used
        "bundle_bytes": sum(entry.byte_count for entry in manifest.files), "bundle_files": {entry.file_name: entry.byte_count for entry in manifest.files},  # bundle size in total and per file
        "chapter_alignment_by_book_index": alignment_report, "passages": describe_passage_tokens(passages),  # alignment per book and passage statistics
        "pairs": describe_pair_chapter_rates(pairs, placement_counts),  # pair placement statistics
        "encoder_parity": json.loads(parity_path.read_text(encoding="utf-8")) if parity_path.is_file() else "not measured in this build",  # written by encoder_parity_check
        "stage_seconds": {name: round(seconds, 1) for name, seconds in stage_seconds.items()},  # time per stage
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)  # the report folder
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")  # readable, stable
