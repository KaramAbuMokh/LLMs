# The builder must reproduce the shared fixture bundle: same lexical arrays, same vectors, a manifest that passes its own check, the fixture index version
import dataclasses  # settings with the fixture's chunk sizes
from pathlib import Path  # temporary bundle folder
import numpy  # array comparison
from retrieval_contracts.bundle_manifest import (  # file names and manifest helpers
    CHAPTER_RECORDS_FILE_NAME, MANIFEST_FILE_NAME, PAIR_LEXICAL_FILE_NAME, PAIR_RECORDS_FILE_NAME, PAIR_VECTORS_FILE_NAME, PASSAGE_LEXICAL_FILE_NAME,  # file names
    PASSAGE_RECORDS_FILE_NAME, PASSAGE_VECTORS_FILE_NAME, find_bundle_problems, manifest_from_json,  # more file names and manifest helpers
)
from retrieval_contracts.bundle_records import ChapterRecord, PassageRecord, QuestionAnswerRecord, read_json_lines  # stored records
from retrieval_contracts.shared_fixture.fixture_encoders import HashingTextEncoder  # fake encoder
from retrieval_contracts.shared_fixture.fixture_files import FIXTURE_BUNDLE_FOLDER  # reference bundle
from retrieval_index_builder.bundle_writing.bundle_assembly import compose_index_version, write_index_bundle  # code under test
from retrieval_index_builder.index_build_settings import build_default_settings  # base settings

FIXTURE_BUILT_AT = "2026-10-24T00:00:00Z"  # build time stored in the fixture manifest


def write_fixture_bundle(output_folder: Path):  # builds a bundle from the fixture's records with the hashing encoder
    settings = dataclasses.replace(build_default_settings(), target_tokens=40, max_tokens=60, overlap_sentences=0, embedding_batch_size=2)  # the fixture's chunking
    passages = read_json_lines(FIXTURE_BUNDLE_FOLDER / PASSAGE_RECORDS_FILE_NAME, PassageRecord)  # fixture passages
    pairs = read_json_lines(FIXTURE_BUNDLE_FOLDER / PAIR_RECORDS_FILE_NAME, QuestionAnswerRecord)  # fixture pairs
    chapters = read_json_lines(FIXTURE_BUNDLE_FOLDER / CHAPTER_RECORDS_FILE_NAME, ChapterRecord)  # fixture chapters
    return write_index_bundle(output_folder, passages, pairs, chapters, HashingTextEncoder(), settings, [], FIXTURE_BUILT_AT)  # the manifest


def test_lexical_arrays_equal_the_fixture_exactly(tmp_path):  # BM25 weights, postings and terms reproduce bit for bit
    write_fixture_bundle(tmp_path)  # our bundle
    for file_name in (PASSAGE_LEXICAL_FILE_NAME, PAIR_LEXICAL_FILE_NAME):  # both lexical files
        with numpy.load(tmp_path / file_name, allow_pickle=False) as ours, numpy.load(FIXTURE_BUNDLE_FOLDER / file_name, allow_pickle=False) as reference:  # both the builder's file and the fixture's file, opened without pickles
            assert sorted(ours.files) == sorted(reference.files)  # same four arrays
            for array_name in reference.files:  # each array
                assert ours[array_name].dtype == reference[array_name].dtype  # same type
                assert numpy.array_equal(ours[array_name], reference[array_name])  # same values, no tolerance


def test_vectors_equal_the_fixture_to_one_millionth(tmp_path):  # embeddings reproduce the stored vectors
    write_fixture_bundle(tmp_path)  # our bundle
    for file_name in (PASSAGE_VECTORS_FILE_NAME, PAIR_VECTORS_FILE_NAME):  # both vector files
        ours = numpy.load(tmp_path / file_name, allow_pickle=False)  # our vectors
        reference = numpy.load(FIXTURE_BUNDLE_FOLDER / file_name, allow_pickle=False)  # fixture vectors
        assert ours.dtype == numpy.float32 and ours.shape == reference.shape  # same type and shape
        assert numpy.allclose(ours, reference, atol=1e-6, rtol=0)  # to 1e-6


def test_record_files_are_byte_identical_to_the_fixture(tmp_path):  # the strict JSON-lines writer gives the same bytes
    write_fixture_bundle(tmp_path)  # our bundle
    for file_name in (PASSAGE_RECORDS_FILE_NAME, PAIR_RECORDS_FILE_NAME, CHAPTER_RECORDS_FILE_NAME):  # the three record files
        assert (tmp_path / file_name).read_bytes() == (FIXTURE_BUNDLE_FOLDER / file_name).read_bytes()  # same bytes


def test_own_manifest_passes_the_bundle_check_and_counts_match(tmp_path):  # the manifest describes the folder it sits in
    manifest = write_fixture_bundle(tmp_path)  # our bundle
    reloaded = manifest_from_json((tmp_path / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))  # parsed back strictly
    assert reloaded == manifest  # the file holds what was returned
    assert find_bundle_problems(tmp_path, reloaded) == []  # no problems
    assert (manifest.passage_count, manifest.pair_count, manifest.chapter_count) == (6, 4, 4)  # fixture sizes
    assert manifest.text_encoder == HashingTextEncoder().fingerprint  # the encoder's fingerprint is recorded


def test_index_version_follows_the_rule_and_matches_the_fixture():  # date of the build plus 8 hex of the joined file checksums
    fixture_manifest = manifest_from_json((FIXTURE_BUNDLE_FOLDER / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))  # fixture manifest
    assert compose_index_version(FIXTURE_BUILT_AT, fixture_manifest.files) == "got-retrieval-20261024-eeb6e60a"  # the value of the design
    assert compose_index_version("2027-01-02T03:04:05Z", fixture_manifest.files).startswith("got-retrieval-20270102-")  # date comes from the build time
    assert compose_index_version(FIXTURE_BUILT_AT, fixture_manifest.files[::-1]) != fixture_manifest.index_version  # file order matters
