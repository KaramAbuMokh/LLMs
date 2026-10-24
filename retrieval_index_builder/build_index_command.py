# The build command: python -m retrieval_index_builder.build_index_command [--upload] [--target-tokens N --max-tokens M --output-folder P]
import argparse  # command line
import dataclasses  # settings overrides
import json  # upload receipt
import time  # stage timings
from pathlib import Path  # command line paths
from retrieval_contracts.book_parts import BOOK_TRAIN_PARTS_FILE_NAME, HELD_OUT_PASSAGE_FILE_NAME  # source files that go into the manifest checksums
from retrieval_encoders.pinned_text_encoder import PinnedModelTextEncoder  # the shared dense encoder
from retrieval_index_builder.build_report_writing import write_build_report  # report file
from retrieval_index_builder.bundle_writing.bundle_assembly import write_index_bundle  # the seven files and the manifest
from retrieval_index_builder.bundle_writing.bundle_upload import upload_index_bundle  # private upload
from retrieval_index_builder.index_build_settings import BUILDER_FOLDER, IndexBuildSettings, build_default_settings  # settings
from retrieval_index_builder.passage_chunking.chapter_chunking import chunk_all_parts  # passages
from retrieval_index_builder.source_alignment.book_part_texts import load_book_part_texts  # part texts
from retrieval_index_builder.source_alignment.chapter_alignment import align_chapters  # chapters
from retrieval_index_builder.source_alignment.pair_records import build_pair_records  # pairs with book and chapter
from retrieval_index_builder.source_alignment.heading_reading.raw_chapter_headings import RAW_BOOK_FILE_NAMES, read_raw_chapter_headings  # expected headings

PAIRS_FILE_NAME = "question_answer_train.json"  # the 34,001 cleaned training pairs inside the source data folder
BUILD_OUTPUT_FOLDER = BUILDER_FOLDER / "build_output"  # everything generated lives here


def parse_arguments(arguments: list[str] | None) -> argparse.Namespace:  # the command line
    argument_parser = argparse.ArgumentParser(description="Build the retrieval index bundle.")  # definition
    argument_parser.add_argument("--upload", action="store_true", help="upload the finished bundle to the private index repo")  # real bundle only
    argument_parser.add_argument("--target-tokens", type=int, help="passage target size in GPT-2 tokens")  # ablation builds
    argument_parser.add_argument("--max-tokens", type=int, help="passage ceiling in GPT-2 tokens")  # ablation builds
    argument_parser.add_argument("--output-folder", type=Path, help="bundle folder (ablation builds are never uploaded)")  # default: build_output/bundle
    return argument_parser.parse_args(arguments)  # parsed values


def build_settings_from_arguments(parsed: argparse.Namespace) -> IndexBuildSettings:  # defaults plus the command line overrides
    if parsed.upload and parsed.output_folder:  # an ablation bundle must never reach the index repo
        raise SystemExit("--upload is only for the default bundle folder")  # refuse
    overrides = {name: value for name, value in (("target_tokens", parsed.target_tokens), ("max_tokens", parsed.max_tokens)) if value is not None}  # given values
    return dataclasses.replace(build_default_settings(parsed.output_folder), **overrides)  # frozen settings with the overrides


def run_build(settings: IndexBuildSettings, report_path: Path) -> Path:  # one full build; returns the bundle folder
    stage_seconds, stage_start = {}, time.monotonic()  # timings per stage
    part_texts = load_book_part_texts(settings.source_data_folder)  # seven cleaned part texts
    headings_by_book = {book_index: read_raw_chapter_headings(settings.raw_books_folder / file_name) for book_index, file_name in RAW_BOOK_FILE_NAMES.items()}  # expected headings
    alignment = align_chapters(part_texts, headings_by_book, settings.minimum_aligned_share)  # chapters per part
    stage_seconds["chapter_alignment"], stage_start = time.monotonic() - stage_start, time.monotonic()  # time and restart the clock
    pairs, placement_counts = build_pair_records(settings.source_data_folder / PAIRS_FILE_NAME, part_texts, alignment.chapters, settings)  # pairs with book and chapter
    stage_seconds["pair_placement"], stage_start = time.monotonic() - stage_start, time.monotonic()  # time and restart
    passages = chunk_all_parts(part_texts, alignment.chapters, settings)  # sentence-aligned passages
    stage_seconds["passage_chunking"], stage_start = time.monotonic() - stage_start, time.monotonic()  # time and restart
    encoder = PinnedModelTextEncoder(settings.model_files_folder, thread_count=settings.encoder_thread_count, batch_size=settings.embedding_batch_size)  # dense encoder
    source_files = [settings.source_data_folder / name for name in (PAIRS_FILE_NAME, BOOK_TRAIN_PARTS_FILE_NAME, HELD_OUT_PASSAGE_FILE_NAME)]  # checksummed inputs
    manifest = write_index_bundle(settings.output_folder, passages, pairs, alignment.chapters, encoder, settings, source_files)  # vectors, postings, manifest
    stage_seconds["embedding_and_bundle_writing"] = time.monotonic() - stage_start  # time of the heaviest stage
    write_build_report(report_path, manifest, alignment.report, passages, pairs, placement_counts, stage_seconds, BUILD_OUTPUT_FOLDER / "encoder_parity.json")  # report
    print("built %s: %d passages, %d pairs, %d chapters" % (manifest.index_version, len(passages), len(pairs), len(alignment.chapters)))  # one summary line
    return settings.output_folder  # where the bundle is


def main(arguments: list[str] | None = None) -> None:  # entry point
    parsed = parse_arguments(arguments)  # command line
    settings = build_settings_from_arguments(parsed)  # settings
    report_path = (settings.output_folder / "build_report.json") if parsed.output_folder else BUILD_OUTPUT_FOLDER / "build_report.json"  # ablations keep their own report
    bundle_folder = run_build(settings, report_path)  # build
    if parsed.upload:  # real bundle only
        receipt = upload_index_bundle(bundle_folder, settings.index_repo_id, settings.token_file.read_text(encoding="utf-8").strip())  # token read here, passed on, never printed
        (BUILD_OUTPUT_FOLDER / "upload_receipt.json").write_text(json.dumps(dataclasses.asdict(receipt), indent=2) + "\n", encoding="utf-8")  # repo, commit, version
        print("uploaded %s at commit %s" % (receipt.repo_id, receipt.commit_hash[:8]))  # no token anywhere


if __name__ == "__main__":  # run as a command
    main()  # start the build
