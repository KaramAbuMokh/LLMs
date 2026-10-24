# The build report: passage statistics, pair chapter rates per book, and the written JSON (no book text in it)
import json  # report file
from retrieval_contracts.bundle_manifest import MANIFEST_FILE_NAME, manifest_from_json  # fixture manifest
from retrieval_contracts.bundle_records import PassageRecord, QuestionAnswerRecord  # records to summarise
from retrieval_contracts.shared_fixture.fixture_files import FIXTURE_BUNDLE_FOLDER  # a valid manifest
from retrieval_index_builder.build_report_writing import describe_pair_chapter_rates, describe_passage_tokens, write_build_report  # code under test


def make_passage(token_count: int, chapter_number: int | None) -> PassageRecord:  # a passage record with a given size
    return PassageRecord("bp-1-000-000", 1, "Book", 1, chapter_number, None, 0, 10, token_count, "text")  # minimal record


def make_pair(pair_id: int, book_title: str | None, chapter_number: int | None) -> QuestionAnswerRecord:  # a pair record with a given place
    return QuestionAnswerRecord("qa-%d" % pair_id, pair_id, "q", "a", book_title, 1 if book_title else None, chapter_number, None)  # minimal record


def test_passage_statistics_use_nearest_rank_percentiles_and_count_chapterless_passages():  # sizes 1..100
    passages = [make_passage(size, None if size > 90 else 1) for size in range(1, 101)]  # 100 passages, the last 10 without chapter
    statistics = describe_passage_tokens(passages)  # code under test
    assert (statistics["passage_count"], statistics["minimum_tokens"], statistics["maximum_tokens"], statistics["total_tokens"]) == (100, 1, 100, 5050)  # totals
    assert (statistics["p5"], statistics["p50"], statistics["p95"]) == (6, 51, 96)  # nearest-rank percentiles
    assert statistics["passages_without_chapter"] == 10  # appendix-like passages


def test_chapter_rates_are_computed_per_book_and_overall():  # 3 pairs of book A (2 with chapter), 1 of book B (none), 2 unplaced
    pairs = [make_pair(1, "A", 3), make_pair(2, "A", 4), make_pair(3, "A", None), make_pair(4, "B", None), make_pair(5, None, None), make_pair(6, None, None)]  # six pairs
    rates = describe_pair_chapter_rates(pairs, {"shingles_in_window": 4, "unplaced": 2})  # code under test
    assert rates["pairs_with_chapter"] == 2 and rates["chapter_rate"] == round(2 / 6, 4)  # overall
    assert rates["chapter_rate_by_book"] == {"A": round(2 / 3, 4), "B": 0.0, "unplaced": 0.0}  # per book
    assert rates["pairs_by_book"] == {"A": 3, "B": 1, "unplaced": 2} and rates["placement_counts_by_method"]["unplaced"] == 2  # counts


def test_written_report_has_every_section_and_no_passage_text(tmp_path):  # the file other agents read
    manifest = manifest_from_json((FIXTURE_BUNDLE_FOLDER / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))  # fixture manifest
    report_path = tmp_path / "reports" / "build_report.json"  # report location
    secret_text = "SECRET-PASSAGE-WORDS"  # stands for book text
    passage = PassageRecord("bp-1-000-000", 1, "Book", 1, 1, "Arya", 0, 10, 12, secret_text)  # a passage carrying text
    write_build_report(report_path, manifest, {"book_1": {"gate_passed": True}}, [passage], [make_pair(1, "Book", 1)], {"shingles_in_window": 1}, {"chapter_alignment": 1.234}, tmp_path / "missing.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))  # parsed back
    assert {"index_version", "chapter_alignment_by_book_index", "passages", "pairs", "encoder_parity", "stage_seconds", "bundle_bytes"} <= set(report)  # sections
    assert report["encoder_parity"] == "not measured in this build" and report["stage_seconds"] == {"chapter_alignment": 1.2}  # defaults and rounding
    assert secret_text not in report_path.read_text(encoding="utf-8")  # no book text in the report
