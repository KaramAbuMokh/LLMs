# Reads a downloaded retrieval bundle into memory after checking everything the manifest promises (checksums, versions, encoder, counts)
from dataclasses import dataclass  # frozen result records
from pathlib import Path  # bundle folder
from typing import Mapping  # term table type
import numpy  # arrays
from numpy.typing import NDArray  # array type hints
from retrieval_contracts.bundle_manifest import (  # file names and manifest helpers
    CHAPTER_RECORDS_FILE_NAME, LEXICAL_ARRAY_NAMES, MANIFEST_FILE_NAME, PAIR_LEXICAL_FILE_NAME, PAIR_RECORDS_FILE_NAME,
    PAIR_VECTORS_FILE_NAME, PASSAGE_LEXICAL_FILE_NAME, PASSAGE_RECORDS_FILE_NAME, PASSAGE_VECTORS_FILE_NAME, BundleManifest,
    find_bundle_problems, manifest_from_json,
)
from retrieval_contracts.bundle_records import ChapterRecord, PassageRecord, QuestionAnswerRecord, read_json_lines  # stored records
from retrieval_contracts.encoder_contracts import EncoderFingerprint  # which encoder made the vectors
from retrieval_contracts.indexed_text import INDEXED_TEXT_VERSION  # version of the indexed-text rules
from retrieval_contracts.lexical_tokens import LEXICAL_TOKENIZER_VERSION  # version of the word splitter


class BundleValidationError(Exception):  # the bundle cannot be trusted; the message lists reasons (file names and numbers, never text)
    pass


@dataclass(frozen=True)
class LexicalPostings:  # lexical index of one corpus: for every term the documents holding it with their full BM25 weights
    term_rows: Mapping[str, int]  # term -> its position in the posting_starts table
    posting_starts: NDArray[numpy.int64]  # term i owns postings [starts[i], starts[i + 1])
    posting_rows: NDArray[numpy.int32]  # document row of each posting, ascending within a term
    posting_weights: NDArray[numpy.float32]  # BM25 weight of the term in that document
    document_count: int  # documents of the corpus (rows of the matching vector file)


@dataclass(frozen=True)
class LoadedIndexBundle:  # a validated bundle held read-only in memory
    manifest: BundleManifest  # what the bundle says about itself
    passages: tuple[PassageRecord, ...]  # row i = passage vector i
    passage_vectors: NDArray[numpy.float32]  # (passages, dimensions), unit rows
    passage_lexical: LexicalPostings  # lexical index over the indexed passage texts
    pairs: tuple[QuestionAnswerRecord, ...]  # row i = question vector i
    question_vectors: NDArray[numpy.float32]  # (pairs, dimensions), unit rows of the question alone
    pair_lexical: LexicalPostings  # lexical index over question + answer
    chapters: tuple[ChapterRecord, ...]  # chapter spans of the book parts


def load_lexical_postings(npz_path: Path, document_count: int) -> LexicalPostings:  # one lexical .npz (no pickles) as postings
    with numpy.load(npz_path, allow_pickle=False) as stored_arrays:  # refuse pickled objects
        if set(stored_arrays.files) != set(LEXICAL_ARRAY_NAMES):  # a different layout than the contract
            raise BundleValidationError("%s has arrays %s" % (npz_path.name, sorted(stored_arrays.files)))  # names only
        terms, starts = stored_arrays["terms"], stored_arrays["posting_starts"]  # term table and offsets
        rows, weights = stored_arrays["posting_rows"], stored_arrays["posting_weights"]  # the postings
    if len(starts) != len(terms) + 1 or starts[-1] != len(rows) or len(rows) != len(weights):  # arrays disagree about their size
        raise BundleValidationError("%s has inconsistent posting arrays" % npz_path.name)  # refuse
    return LexicalPostings({term: position for position, term in enumerate(terms.tolist())}, starts, rows, weights, document_count)  # frozen view


def find_manifest_problems(manifest: BundleManifest, text_encoder_fingerprint: EncoderFingerprint) -> list[str]:  # reasons this code cannot use the bundle
    problems = []  # human-readable findings
    if manifest.lexical.tokenizer_version != LEXICAL_TOKENIZER_VERSION:  # words would be split differently than at build time
        problems.append("lexical tokenizer version %d, this code uses %d" % (manifest.lexical.tokenizer_version, LEXICAL_TOKENIZER_VERSION))  # refuse
    if manifest.chunking.indexed_text_version != INDEXED_TEXT_VERSION:  # headings would be composed differently
        problems.append("indexed text version %d, this code uses %d" % (manifest.chunking.indexed_text_version, INDEXED_TEXT_VERSION))  # refuse
    if manifest.text_encoder != text_encoder_fingerprint:  # questions would be embedded by another model than the passages were
        problems.append("text encoder fingerprint differs from the loaded encoder")  # refuse
    return problems  # empty when compatible


def read_array(bundle_folder: Path, file_name: str) -> NDArray[numpy.float32]:  # one vector file, read-only
    vectors = numpy.load(bundle_folder / file_name, allow_pickle=False)  # refuse pickled objects
    vectors.flags.writeable = False  # nothing may change the index while serving
    return vectors  # the matrix


def load_index_bundle(bundle_folder: Path, text_encoder_fingerprint: EncoderFingerprint) -> LoadedIndexBundle:  # validates, then loads
    manifest_path = bundle_folder / MANIFEST_FILE_NAME  # the file written last by the builder
    if not manifest_path.is_file():  # nothing downloaded
        raise BundleValidationError("%s is missing" % MANIFEST_FILE_NAME)  # refuse
    try:  # a bad manifest must become a validation error, not a crash
        manifest = manifest_from_json(manifest_path.read_text(encoding="utf-8"))  # strict parse
    except (ValueError, KeyError, TypeError) as error:  # unknown keys, bad JSON or missing sections
        raise BundleValidationError("%s is unreadable (%s)" % (MANIFEST_FILE_NAME, type(error).__name__)) from error  # type name only
    problems = [*find_bundle_problems(bundle_folder, manifest), *find_manifest_problems(manifest, text_encoder_fingerprint)]  # all findings
    if problems:  # checksums, versions or encoder disagree
        raise BundleValidationError("; ".join(problems))  # list every reason
    try:  # stored records are parsed strictly
        passages = read_json_lines(bundle_folder / PASSAGE_RECORDS_FILE_NAME, PassageRecord)  # passage rows
        pairs = read_json_lines(bundle_folder / PAIR_RECORDS_FILE_NAME, QuestionAnswerRecord)  # pair rows
        chapters = read_json_lines(bundle_folder / CHAPTER_RECORDS_FILE_NAME, ChapterRecord)  # chapter spans
    except ValueError as error:  # a record has unknown or missing keys
        raise BundleValidationError("a records file does not match the contract (%s)" % type(error).__name__) from error  # type name only
    passage_vectors = read_array(bundle_folder, PASSAGE_VECTORS_FILE_NAME)  # passage matrix
    question_vectors = read_array(bundle_folder, PAIR_VECTORS_FILE_NAME)  # question matrix
    expected_shapes = ((len(passages), manifest.passage_count, passage_vectors), (len(pairs), manifest.pair_count, question_vectors))  # records vs manifest vs matrix
    if len(chapters) != manifest.chapter_count or any(  # counts the manifest promised
            record_count != promised_count or vectors.shape != (record_count, text_encoder_fingerprint.output_dimensions)
            for record_count, promised_count, vectors in expected_shapes):
        raise BundleValidationError("record counts or vector shapes differ from the manifest")  # numbers only
    return LoadedIndexBundle(  # everything checked
        manifest, passages, passage_vectors, load_lexical_postings(bundle_folder / PASSAGE_LEXICAL_FILE_NAME, len(passages)),
        pairs, question_vectors, load_lexical_postings(bundle_folder / PAIR_LEXICAL_FILE_NAME, len(pairs)), chapters,
    )
