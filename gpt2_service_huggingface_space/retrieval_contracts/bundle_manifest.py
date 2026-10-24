# The bundle manifest: format version, file names, what the index was built from and with, and a checksum per file
import hashlib  # file checksums
import json  # manifest text
from dataclasses import asdict, dataclass  # frozen manifest parts
from pathlib import Path  # bundle folder paths
from retrieval_contracts.bundle_records import build_strict_record  # strict rebuild of stored dataclasses
from retrieval_contracts.encoder_contracts import EncoderFingerprint  # which encoder made the vectors

BUNDLE_FORMAT_VERSION = 1  # bump on any change to the files or their fields; the service refuses versions it does not know
MANIFEST_FILE_NAME = "manifest.json"  # written last by the builder, read first by the service
PASSAGE_RECORDS_FILE_NAME = "book_passages.jsonl"  # one PassageRecord per line
PASSAGE_VECTORS_FILE_NAME = "book_passage_vectors.npy"  # float32 (passages, dimensions), unit-length rows, row i = line i
PASSAGE_LEXICAL_FILE_NAME = "book_passage_lexical_index.npz"  # lexical postings over compose_indexed_passage_text
PAIR_RECORDS_FILE_NAME = "question_answer_pairs.jsonl"  # one QuestionAnswerRecord per line
PAIR_VECTORS_FILE_NAME = "question_vectors.npy"  # float32 (pairs, dimensions) of the question text alone
PAIR_LEXICAL_FILE_NAME = "question_answer_lexical_index.npz"  # lexical postings over compose_indexed_pair_text
CHAPTER_RECORDS_FILE_NAME = "chapters.jsonl"  # one ChapterRecord per line
BUNDLE_DATA_FILE_NAMES = (  # every file except the manifest, in manifest order
    PASSAGE_RECORDS_FILE_NAME, PASSAGE_VECTORS_FILE_NAME, PASSAGE_LEXICAL_FILE_NAME,
    PAIR_RECORDS_FILE_NAME, PAIR_VECTORS_FILE_NAME, PAIR_LEXICAL_FILE_NAME, CHAPTER_RECORDS_FILE_NAME,
)
LEXICAL_ARRAY_NAMES = ("terms", "posting_starts", "posting_rows", "posting_weights")  # the four arrays inside each lexical .npz (DESIGN.md §6.3)


@dataclass(frozen=True)
class ChunkingSpecification:  # how the book text was cut into passages
    chunker_version: int  # version of the index builder's chunking code
    target_tokens: int  # GPT-2 tokens a passage aims for
    max_tokens: int  # hard ceiling per passage
    overlap_sentences: int  # sentences repeated from the end of the previous passage of the same chapter
    indexed_text_version: int  # INDEXED_TEXT_VERSION used for the vectors and postings


@dataclass(frozen=True)
class LexicalSpecification:  # how the lexical postings were weighted
    tokenizer_version: int  # LEXICAL_TOKENIZER_VERSION used on both sides
    saturation_k1: float  # BM25 term-frequency saturation (1.2)
    length_normalization_b: float  # BM25 document-length normalisation (0.75)


@dataclass(frozen=True)
class BundleFileEntry:  # one data file of the bundle
    file_name: str  # name inside the bundle folder
    sha256: str  # hex checksum of its bytes
    byte_count: int  # its size


@dataclass(frozen=True)
class BundleManifest:  # everything the service checks before it trusts a bundle
    bundle_format_version: int  # BUNDLE_FORMAT_VERSION at build time
    index_version: str  # "got-retrieval-<yyyymmdd of built_at_utc>-<first 8 hex of sha256 over the 7 file checksums joined in `files` order>"
    built_at_utc: str  # ISO 8601 build time
    builder_source_sha256: str  # checksum over the builder's own .py files (the project has no git history to point at)
    source_data_sha256: dict[str, str]  # source file name -> checksum (pairs, book parts, held-out passage)
    text_encoder: EncoderFingerprint  # encoder that made both vector files
    chunking: ChunkingSpecification  # passage cutting settings
    lexical: LexicalSpecification  # postings settings
    passage_count: int  # lines in book_passages.jsonl = rows of the passage vectors
    pair_count: int  # lines in question_answer_pairs.jsonl = rows of the question vectors
    chapter_count: int  # lines in chapters.jsonl
    files: tuple[BundleFileEntry, ...]  # one entry per data file, in BUNDLE_DATA_FILE_NAMES order


def manifest_to_json(manifest: BundleManifest) -> str:  # stable, readable manifest text
    return json.dumps(asdict(manifest), indent=2, sort_keys=True, ensure_ascii=False) + "\n"  # sorted keys, so rebuilds diff cleanly


def manifest_from_json(manifest_text: str) -> BundleManifest:  # strict parse; unknown or missing keys raise ValueError
    values = json.loads(manifest_text)  # the stored object
    values["text_encoder"] = build_strict_record(EncoderFingerprint, values["text_encoder"])  # nested encoder fingerprint
    values["chunking"] = build_strict_record(ChunkingSpecification, values["chunking"])  # nested chunking settings
    values["lexical"] = build_strict_record(LexicalSpecification, values["lexical"])  # nested postings settings
    values["files"] = tuple(build_strict_record(BundleFileEntry, entry) for entry in values["files"])  # file entries
    return build_strict_record(BundleManifest, values)  # the frozen manifest


def compute_file_sha256(path: Path) -> str:  # hex checksum of one file, read in 1 MB pieces
    digest = hashlib.sha256()  # running checksum
    with path.open("rb") as input_file:  # raw bytes
        for piece in iter(lambda: input_file.read(1 << 20), b""):  # until the end of the file
            digest.update(piece)  # add the piece
    return digest.hexdigest()  # final hex string


def describe_bundle_file(bundle_folder: Path, file_name: str) -> BundleFileEntry:  # manifest entry for one written file
    path = bundle_folder / file_name  # the file
    return BundleFileEntry(file_name, compute_file_sha256(path), path.stat().st_size)  # name, checksum, size


def find_bundle_problems(bundle_folder: Path, manifest: BundleManifest) -> list[str]:  # empty list = the folder matches its manifest
    problems = []  # human-readable findings (file names and numbers only)
    if manifest.bundle_format_version != BUNDLE_FORMAT_VERSION:  # written by another format version
        problems.append("bundle format %d, this code reads %d" % (manifest.bundle_format_version, BUNDLE_FORMAT_VERSION))  # refuse
    if tuple(entry.file_name for entry in manifest.files) != BUNDLE_DATA_FILE_NAMES:  # missing, extra or reordered entries
        problems.append("manifest lists %s" % [entry.file_name for entry in manifest.files])  # refuse
    for entry in manifest.files:  # every listed file
        path = bundle_folder / entry.file_name  # where it should be
        if not path.is_file():  # not downloaded or not written
            problems.append("%s is missing" % entry.file_name)  # refuse
        elif path.stat().st_size != entry.byte_count or compute_file_sha256(path) != entry.sha256:  # changed or cut short
            problems.append("%s does not match its checksum" % entry.file_name)  # refuse
    return problems  # the caller decides (the service turns retrieval off, the builder fails the build)
