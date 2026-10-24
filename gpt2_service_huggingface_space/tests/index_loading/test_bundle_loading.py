# Tests of the bundle loader: it accepts the fixture and refuses a changed byte, a wrong encoder, a wrong tokenizer version and wrong counts
import json  # manifest edits
import shutil  # copies the fixture bundle
import pytest  # expected errors
from gpt2_demo_service.index_loading.bundle_loading import BundleValidationError, load_index_bundle  # code under test
from retrieval_contracts.bundle_manifest import MANIFEST_FILE_NAME  # manifest file name
from retrieval_contracts.shared_fixture.fixture_encoders import HASHING_ENCODER_FINGERPRINT  # the matching fingerprint
from retrieval_contracts.shared_fixture.fixture_files import FIXTURE_BUNDLE_FOLDER  # the reference bundle


@pytest.fixture()
def bundle_copy(tmp_path):  # a writable copy of the fixture bundle
    destination = tmp_path / "bundle"  # where the copy lives
    shutil.copytree(FIXTURE_BUNDLE_FOLDER, destination)  # copy every file
    return destination  # its folder


def edit_manifest(bundle_folder, edit):  # rewrites the manifest after the given edit (the manifest carries no checksum of itself)
    manifest_path = bundle_folder / MANIFEST_FILE_NAME  # the file
    values = json.loads(manifest_path.read_text(encoding="utf-8"))  # parsed
    edit(values)  # change it
    manifest_path.write_text(json.dumps(values), encoding="utf-8")  # stored again


def test_fixture_bundle_loads(fixture_bundle):  # counts, shapes and read-only arrays
    assert (len(fixture_bundle.passages), len(fixture_bundle.pairs), len(fixture_bundle.chapters)) == (6, 4, 4)  # record counts
    assert fixture_bundle.passage_vectors.shape == (6, 256) and fixture_bundle.question_vectors.shape == (4, 256)  # matrix shapes
    assert not fixture_bundle.passage_vectors.flags.writeable  # nothing may change the index while serving
    assert fixture_bundle.passage_lexical.document_count == 6 and fixture_bundle.pair_lexical.document_count == 4  # corpus sizes


def test_changed_byte_is_refused_naming_the_file(bundle_copy):  # a corrupted download must never load
    vectors_path = bundle_copy / "book_passage_vectors.npy"  # a data file
    content = bytearray(vectors_path.read_bytes())  # its bytes
    content[-1] ^= 0xFF  # flip the last byte
    vectors_path.write_bytes(bytes(content))  # same size, different content
    with pytest.raises(BundleValidationError, match="book_passage_vectors.npy"):  # the reason names the file
        load_index_bundle(bundle_copy, HASHING_ENCODER_FINGERPRINT)  # load


def test_wrong_encoder_fingerprint_is_refused(bundle_copy):  # vectors from another model would give nonsense scores
    other_fingerprint = HASHING_ENCODER_FINGERPRINT.__class__(**{**HASHING_ENCODER_FINGERPRINT.__dict__, "revision": "another-revision"})  # one field differs
    with pytest.raises(BundleValidationError, match="fingerprint"):  # named reason
        load_index_bundle(bundle_copy, other_fingerprint)  # load


def test_wrong_lexical_tokenizer_version_is_refused(bundle_copy):  # words would be split differently than at build time
    edit_manifest(bundle_copy, lambda values: values["lexical"].update(tokenizer_version=99))  # pretend another version
    with pytest.raises(BundleValidationError, match="lexical tokenizer version"):  # named reason
        load_index_bundle(bundle_copy, HASHING_ENCODER_FINGERPRINT)  # load


def test_wrong_indexed_text_version_is_refused(bundle_copy):  # headings would be composed differently
    edit_manifest(bundle_copy, lambda values: values["chunking"].update(indexed_text_version=99))  # pretend another version
    with pytest.raises(BundleValidationError, match="indexed text version"):  # named reason
        load_index_bundle(bundle_copy, HASHING_ENCODER_FINGERPRINT)  # load


def test_count_mismatch_is_refused(bundle_copy):  # the manifest must agree with the files
    edit_manifest(bundle_copy, lambda values: values.update(passage_count=7))  # promise one passage too many
    with pytest.raises(BundleValidationError, match="counts"):  # named reason
        load_index_bundle(bundle_copy, HASHING_ENCODER_FINGERPRINT)  # load


def test_missing_manifest_is_refused(tmp_path):  # an empty folder is not a bundle
    with pytest.raises(BundleValidationError, match="manifest.json"):  # named reason
        load_index_bundle(tmp_path, HASHING_ENCODER_FINGERPRINT)  # load
