# Tests of retrieval start-up: a good bundle gives a ready status and the copy guard's index, every failure turns retrieval OFF with a reason
from dataclasses import replace  # copies settings with changes
from gpt2_demo_service.index_loading.bundle_loading import BundleValidationError  # error type the loader raises
from gpt2_demo_service.service_startup.retrieval_startup import assemble_retrieval  # code under test
from retrieval_contracts.shared_fixture.fixture_encoders import HASHING_ENCODER_FINGERPRINT, HashingTextEncoder  # fakes
from retrieval_contracts.shared_fixture.fixture_files import FIXTURE_BUNDLE_FOLDER  # reference bundle

def test_good_bundle_gives_a_ready_status(fixture_search_settings):  # counts, version and short revision
    result = assemble_retrieval(FIXTURE_BUNDLE_FOLDER, HashingTextEncoder(), None, fixture_search_settings, "0123456789abcdef" * 2 + "01234567", "passages_only")  # start retrieval over the fixture bundle
    assert result.retriever is not None and result.status.ready and result.status.problem is None  # working
    assert (result.status.passages, result.status.pairs, result.status.index_revision) == (6, 4, "01234567")  # numbers and 8-hex revision
    assert result.book_run_index is not None and result.status.book_runs == result.book_run_index.run_count  # the copy guard has the books
    assert result.status.index_version == "got-retrieval-20261024-eeb6e60a" and result.status.text_encoder == HASHING_ENCODER_FINGERPRINT.model_id  # identity


def test_changed_byte_turns_retrieval_off_naming_the_file(tampered_bundle_folder, fixture_search_settings):  # checksum failure
    result = assemble_retrieval(tampered_bundle_folder, HashingTextEncoder(), None, fixture_search_settings, "a" * 40, "passages_only")  # start retrieval over the broken bundle
    assert result.retriever is None and result.book_run_index is None and not result.status.ready and "question_vectors.npy" in result.status.problem  # reason names the file


def test_wrong_encoder_turns_retrieval_off(fixture_search_settings):  # vectors from another model
    class OtherEncoder(HashingTextEncoder):  # same behaviour, other identity
        @property
        def fingerprint(self):  # different revision
            return replace(HASHING_ENCODER_FINGERPRINT, revision="other")  # an encoder identity that differs
    result = assemble_retrieval(FIXTURE_BUNDLE_FOLDER, OtherEncoder(), None, fixture_search_settings, "a" * 40, "passages_only")  # start retrieval with the wrong encoder
    assert result.retriever is None and "fingerprint" in result.status.problem  # refused


def test_unexpected_error_is_reported_by_type_only(fixture_search_settings):  # no text in the problem
    class BrokenEncoder(HashingTextEncoder):  # fails when asked to encode
        def encode_texts(self, texts):  # simulated model trouble
            raise RuntimeError("secret question text")  # simulated model trouble with a secret text
    result = assemble_retrieval(FIXTURE_BUNDLE_FOLDER, BrokenEncoder(), None, fixture_search_settings, "a" * 40, "passages_only")  # start retrieval with a failing encoder
    assert result.status.problem == "retrieval start-up failed (RuntimeError)" and "secret" not in result.status.problem  # type name only
