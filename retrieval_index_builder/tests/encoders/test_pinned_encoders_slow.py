# Real-model checks (run once on purpose: pytest -m slow): fingerprint, unit rows, batch independence, the reranker's input cap, and the reference parity
import numpy  # vector comparison
import pytest  # markers and skips
from retrieval_contracts.encoder_contracts import PASSAGE_PAIR_SCORER_FILES, PASSAGE_QUERY_INSTRUCTION, TEXT_ENCODER_FILES  # pinned models
from retrieval_encoders.pinned_pair_scorer import PinnedModelPairScorer  # reranker
from retrieval_encoders.pinned_text_encoder import PinnedModelTextEncoder  # dense encoder
from retrieval_index_builder.encoder_parity_check import measure_encoder_parity  # reference comparison
from retrieval_index_builder.index_build_settings import build_default_settings  # model files location

MODEL_FILES_FOLDER = build_default_settings().model_files_folder  # downloaded models
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not MODEL_FILES_FOLDER.is_dir(), reason="model files are not downloaded")]  # whole file


def test_fingerprint_names_the_pinned_commit_and_the_onnx_checksum():  # what the manifest stores
    fingerprint = PinnedModelTextEncoder(MODEL_FILES_FOLDER).fingerprint  # read from the loaded encoder
    assert (fingerprint.model_id, fingerprint.revision) == (TEXT_ENCODER_FILES.model_id, TEXT_ENCODER_FILES.revision)  # pinned commit
    assert len(fingerprint.model_file_sha256) == 64 and fingerprint.output_dimensions == 384 and fingerprint.pooling == "cls"  # checksum, width, pooling
    assert fingerprint.passage_query_instruction == PASSAGE_QUERY_INSTRUCTION  # instruction is part of the fingerprint


def test_rows_are_unit_length_in_input_order_and_independent_of_the_batch_layout():  # sorting by length inside the encoder must not leak out
    texts = ["Who forged the blade?", "A long invented sentence about a harbor town before dawn. " * 8, "Short.", ""]  # mixed lengths and an empty text
    batched = PinnedModelTextEncoder(MODEL_FILES_FOLDER, batch_size=2).encode_texts(texts)  # two batches
    single_encoder = PinnedModelTextEncoder(MODEL_FILES_FOLDER, batch_size=1)  # one text per call
    one_by_one = numpy.concatenate([single_encoder.encode_texts([text]) for text in texts])  # same texts alone
    assert numpy.allclose(numpy.linalg.norm(batched, axis=1), 1.0, atol=1e-5)  # unit rows
    assert numpy.allclose(batched, one_by_one, atol=1e-5)  # padding and order do not change the vectors
    assert PinnedModelTextEncoder(MODEL_FILES_FOLDER).encode_texts([]).shape == (0, 384)  # empty input


def test_scorer_ignores_everything_after_its_input_cap():  # DESIGN.md 8.1: the pair is cut at input_token_limit word pieces
    question = "Who forged the blade Dawnlight?"  # short question
    passage = "Mira Vell forged the blade Dawnlight at Kettle Ford. " + "The harbor was quiet. " * 60  # long passage
    capped_scorer = PinnedModelPairScorer(MODEL_FILES_FOLDER, input_token_limit=32)  # cap at 32 word pieces
    plain_score = capped_scorer.score_pairs(question, [passage])[0]  # score of the long passage
    longer_score = capped_scorer.score_pairs(question, [passage + " Extra words that come far past the cap. " * 20])[0]  # even longer tail
    assert plain_score == pytest.approx(longer_score, abs=1e-4)  # the tail beyond the cap changes nothing
    wide_score = PinnedModelPairScorer(MODEL_FILES_FOLDER, input_token_limit=256).score_pairs(question, [passage])[0]  # a wider cap sees more text
    assert wide_score != pytest.approx(plain_score, abs=1e-4)  # the cap really is applied
    with pytest.raises(ValueError):  # the graph cannot read more than 512
        PinnedModelPairScorer(MODEL_FILES_FOLDER, input_token_limit=PASSAGE_PAIR_SCORER_FILES.max_input_tokens + 1)  # code under test


def test_vectors_and_scores_agree_with_the_reference_library():  # parity (needs sentence-transformers, only in the scratch venv)
    pytest.importorskip("sentence_transformers")  # reference library
    parity = measure_encoder_parity(MODEL_FILES_FOLDER)  # code under test
    assert parity["text_encoder_passed"] and parity["scorer_order_matches"]  # cosine >= 0.9999 and the same passage order
