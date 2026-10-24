# The cross-encoder reranker (ms-marco MiniLM, ONNX): reads a question and a passage together and gives one raw score per passage
from pathlib import Path  # base folder of the model files
from typing import Sequence  # passages in
import numpy  # scores
from numpy.typing import NDArray  # result type
from retrieval_contracts.encoder_contracts import PASSAGE_PAIR_SCORER_FILES, PinnedModelFiles  # pinned model
from retrieval_encoders.model_file_download import model_folder_for  # where the downloaded files sit
from retrieval_encoders.onnx_session_building import build_inference_session  # CPU session
from retrieval_encoders.word_piece_tokenizing import build_word_piece_tokenizer, pad_encodings_into_inputs  # tokenizing and padding


class PinnedModelPairScorer:  # implements the PassagePairScorer protocol of the contract
    def __init__(self, base_folder: Path, input_token_limit: int, files: PinnedModelFiles = PASSAGE_PAIR_SCORER_FILES, thread_count: int = 2) -> None:  # loads the reranker graph and its tokenizer
        model_folder = model_folder_for(files, base_folder)  # this model's folder
        self._session = build_inference_session(model_folder / files.model_file, thread_count)  # loads the graph once
        self._tokenizer = build_word_piece_tokenizer(model_folder, files, input_token_limit)  # cuts each (question, passage) at the limit
        self._declared_input_names = tuple(model_input.name for model_input in self._session.get_inputs())  # inputs the graph wants

    def score_pairs(self, question: str, passage_texts: Sequence[str]) -> NDArray[numpy.float32]:  # one raw logit per passage, higher fits better
        if not passage_texts:  # nothing to score
            return numpy.zeros(0, dtype=numpy.float32)  # empty result
        encodings = self._tokenizer.encode_batch([(question, passage_text) for passage_text in passage_texts])  # pair encoding with segment ids
        graph_inputs = pad_encodings_into_inputs(encodings, self._declared_input_names)  # one padded batch (a handful of passages)
        logits = self._session.run(None, graph_inputs)[0]  # (passages, 1)
        return logits.reshape(len(passage_texts)).astype(numpy.float32)  # one score per passage in the given order
