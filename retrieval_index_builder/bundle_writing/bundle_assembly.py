# Writes the seven data files and the manifest of one retrieval bundle, then checks the folder against its own manifest
import hashlib  # index version checksum
from datetime import datetime, timezone  # build time
from pathlib import Path  # output folder and source files
from typing import Sequence  # inputs
from retrieval_contracts.bundle_manifest import (  # file names, manifest types and helpers
    BUNDLE_DATA_FILE_NAMES, BUNDLE_FORMAT_VERSION, MANIFEST_FILE_NAME, PAIR_LEXICAL_FILE_NAME, PAIR_RECORDS_FILE_NAME, PAIR_VECTORS_FILE_NAME,  # file names and format version
    PASSAGE_LEXICAL_FILE_NAME, PASSAGE_RECORDS_FILE_NAME, PASSAGE_VECTORS_FILE_NAME, CHAPTER_RECORDS_FILE_NAME,  # more file names
    BundleFileEntry, BundleManifest, ChunkingSpecification, LexicalSpecification, compute_file_sha256, describe_bundle_file, find_bundle_problems, manifest_to_json,  # manifest types and helpers
)
from retrieval_contracts.bundle_records import ChapterRecord, PassageRecord, QuestionAnswerRecord, write_json_lines  # stored records
from retrieval_contracts.encoder_contracts import TextEncoder  # shared encoder interface
from retrieval_contracts.indexed_text import INDEXED_TEXT_VERSION, compose_indexed_pair_text, compose_indexed_passage_text  # what gets embedded and indexed
from retrieval_contracts.lexical_tokens import LEXICAL_TOKENIZER_VERSION  # stored in the manifest
from retrieval_index_builder.bundle_writing.builder_source_checksum import compute_builder_source_sha256  # builder checksum
from retrieval_index_builder.bundle_writing.lexical_postings_writing import build_lexical_postings  # BM25 postings
from retrieval_index_builder.bundle_writing.vector_writing import encode_in_batches, save_lexical_postings, save_vectors  # vectors and postings files
from retrieval_index_builder.index_build_settings import IndexBuildSettings  # sizes and batch settings
from retrieval_index_builder.passage_chunking.chapter_chunking import CHUNKER_VERSION  # chunking rules version

SATURATION_K1 = 1.2  # BM25 term-frequency saturation (D6)
LENGTH_NORMALIZATION_B = 0.75  # BM25 document-length normalisation (D6)


def compose_index_version(built_at_utc: str, files: Sequence[BundleFileEntry]) -> str:  # "got-retrieval-<yyyymmdd>-<8 hex>" by the rule of DESIGN.md 6.3
    combined_checksums = "".join(entry.sha256 for entry in files)  # the seven file checksums joined with no separator, in manifest order
    short_digest = hashlib.sha256(combined_checksums.encode("ascii")).hexdigest()[:8]  # first 8 hex characters
    return "got-retrieval-%s-%s" % (built_at_utc[:10].replace("-", ""), short_digest)  # date of the build plus the content digest


def write_index_bundle(output_folder: Path, passages: Sequence[PassageRecord], pairs: Sequence[QuestionAnswerRecord], chapters: Sequence[ChapterRecord],  # writes the whole bundle in the order the format requires
                       encoder: TextEncoder, settings: IndexBuildSettings, source_files: Sequence[Path], built_at_utc: str | None = None) -> BundleManifest:  # encoder, sizes, checksummed source files and an optional fixed build time
    output_folder.mkdir(parents=True, exist_ok=True)  # the bundle folder
    (output_folder / MANIFEST_FILE_NAME).unlink(missing_ok=True)  # an interrupted build must never leave an old manifest beside new files
    passage_texts = [compose_indexed_passage_text(record) for record in passages]  # heading + text, what vectors and postings read
    write_json_lines(output_folder / PASSAGE_RECORDS_FILE_NAME, passages)  # passage records, row order = vector order
    save_vectors(output_folder / PASSAGE_VECTORS_FILE_NAME, encode_in_batches(encoder, passage_texts, settings.embedding_batch_size))  # passage vectors
    save_lexical_postings(output_folder / PASSAGE_LEXICAL_FILE_NAME, build_lexical_postings(passage_texts, SATURATION_K1, LENGTH_NORMALIZATION_B))  # passage postings
    write_json_lines(output_folder / PAIR_RECORDS_FILE_NAME, pairs)  # pair records
    save_vectors(output_folder / PAIR_VECTORS_FILE_NAME, encode_in_batches(encoder, [record.question for record in pairs], settings.embedding_batch_size))  # question vectors
    pair_texts = [compose_indexed_pair_text(record) for record in pairs]  # question + answer, what the pair postings read
    save_lexical_postings(output_folder / PAIR_LEXICAL_FILE_NAME, build_lexical_postings(pair_texts, SATURATION_K1, LENGTH_NORMALIZATION_B))  # pair postings
    write_json_lines(output_folder / CHAPTER_RECORDS_FILE_NAME, chapters)  # chapter records
    file_entries = tuple(describe_bundle_file(output_folder, file_name) for file_name in BUNDLE_DATA_FILE_NAMES)  # checksum and size of the seven files
    build_time = built_at_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # ISO 8601 UTC
    manifest = BundleManifest(  # everything the service checks before it trusts the bundle
        bundle_format_version=BUNDLE_FORMAT_VERSION, index_version=compose_index_version(build_time, file_entries), built_at_utc=build_time,  # format, version and build time
        builder_source_sha256=compute_builder_source_sha256(), source_data_sha256={path.name: compute_file_sha256(path) for path in source_files},  # builder checksum and source data checksums
        text_encoder=encoder.fingerprint,  # which encoder made the vectors
        chunking=ChunkingSpecification(CHUNKER_VERSION, settings.target_tokens, settings.max_tokens, settings.overlap_sentences, INDEXED_TEXT_VERSION),  # how passages were cut
        lexical=LexicalSpecification(LEXICAL_TOKENIZER_VERSION, SATURATION_K1, LENGTH_NORMALIZATION_B),  # how postings were weighted
        passage_count=len(passages), pair_count=len(pairs), chapter_count=len(chapters), files=file_entries,  # record counts and file checksums
    )
    (output_folder / MANIFEST_FILE_NAME).write_text(manifest_to_json(manifest), encoding="utf-8")  # written last
    bundle_problems = find_bundle_problems(output_folder, manifest)  # the folder must match its own manifest
    if bundle_problems:  # a writing bug
        raise ValueError("bundle does not match its manifest: %s" % bundle_problems)  # file names and reasons only
    return manifest  # the finished bundle's manifest
