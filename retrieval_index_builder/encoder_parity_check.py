# One-off check that our ONNX encoders agree with the sentence-transformers reference (run in a scratch venv; sentence-transformers is not a dependency)
import json  # result file
from pathlib import Path  # model files folder and result file
import numpy  # cosines and orders
from retrieval_contracts.encoder_contracts import PASSAGE_PAIR_SCORER_FILES, TEXT_ENCODER_FILES  # pinned models
from retrieval_encoders.pinned_pair_scorer import PinnedModelPairScorer  # our reranker
from retrieval_encoders.pinned_text_encoder import PinnedModelTextEncoder  # our dense encoder
from retrieval_index_builder.index_build_settings import BUILDER_FOLDER  # where the result is written

PARITY_RESULT_PATH = BUILDER_FOLDER / "build_output" / "encoder_parity.json"  # read by the build report
SAMPLE_TEXTS = [  # invented sentences, no book text
    "The harbor of Lumen was quiet before dawn.", "Who forged the silver blade Dawnlight at Kettle Ford?", "A grey heron on a field of green is the sigil of House Orrin.",  # invented sample texts
    "Represent this sentence for searching relevant passages: what is the sigil of House Orrin?", "x " * 300,  # a long text, cut at the encoder's limit
]
SAMPLE_QUESTION = "Who forged the blade Dawnlight?"  # question for the reranker check
SAMPLE_PASSAGES = [  # five passages from least to most relevant, in scrambled order
    "The weather was cold and grey all week.", "Mira Vell worked the bellows in the smithy at Kettle Ford.", "She forged the silver blade Dawnlight for the heir of House Orrin.",  # invented passages, scrambled
    "Corin kept the tower keys in a cedar box.", "Mira Vell forged the blade Dawnlight.",  # more invented passages
]
MINIMUM_COSINE = 0.9999  # DESIGN.md 8.1
SCORER_INPUT_TOKEN_LIMIT = 256  # the shipped reranker cap


def measure_encoder_parity(model_files_folder: Path) -> dict[str, object]:  # compares our vectors and scores with the reference library's
    from sentence_transformers import CrossEncoder, SentenceTransformer  # only available in the scratch venv
    our_vectors = PinnedModelTextEncoder(model_files_folder).encode_texts(SAMPLE_TEXTS)  # our unit vectors
    reference_model = SentenceTransformer(TEXT_ENCODER_FILES.model_id, revision=TEXT_ENCODER_FILES.revision)  # reference at the same commit
    reference_vectors = reference_model.encode(SAMPLE_TEXTS, normalize_embeddings=True)  # reference unit vectors
    cosines = (our_vectors * reference_vectors).sum(axis=1)  # one cosine per text
    our_scores = PinnedModelPairScorer(model_files_folder, SCORER_INPUT_TOKEN_LIMIT).score_pairs(SAMPLE_QUESTION, SAMPLE_PASSAGES)  # our logits
    reference_scorer = CrossEncoder(PASSAGE_PAIR_SCORER_FILES.model_id, revision=PASSAGE_PAIR_SCORER_FILES.revision, max_length=SCORER_INPUT_TOKEN_LIMIT)  # reference
    reference_scores = reference_scorer.predict([(SAMPLE_QUESTION, passage) for passage in SAMPLE_PASSAGES], activation_fn=None)  # raw logits, same as ours
    return {  # what the build report shows
        "text_encoder_sample_count": len(SAMPLE_TEXTS), "text_encoder_minimum_cosine": round(float(cosines.min()), 6), "text_encoder_passed": bool(cosines.min() >= MINIMUM_COSINE),  # dense encoder result
        "scorer_pair_count": len(SAMPLE_PASSAGES), "scorer_order_matches": bool(numpy.array_equal(numpy.argsort(our_scores), numpy.argsort(reference_scores))),  # reranker order result
        "scorer_maximum_absolute_difference": round(float(numpy.abs(our_scores - reference_scores).max()), 6),  # reranker score difference
        "text_encoder_revision": TEXT_ENCODER_FILES.revision[:8], "scorer_revision": PASSAGE_PAIR_SCORER_FILES.revision[:8],  # which commits were compared
    }


if __name__ == "__main__":  # python -m retrieval_index_builder.encoder_parity_check
    from retrieval_index_builder.index_build_settings import build_default_settings  # model files location
    parity_result = measure_encoder_parity(build_default_settings().model_files_folder)  # run the comparison
    PARITY_RESULT_PATH.write_text(json.dumps(parity_result, indent=2) + "\n", encoding="utf-8")  # read by the next build
    print(json.dumps(parity_result))  # shown to the person running it
