# The dense text encoder (bge-small, ONNX, CLS pooling, unit vectors); the SAME class encodes the index offline and questions online
from pathlib import Path  # base folder of the model files
from typing import Sequence  # texts in
import numpy  # vectors
from numpy.typing import NDArray  # result type
from retrieval_contracts.bundle_manifest import compute_file_sha256  # checksum of the ONNX file
from retrieval_contracts.encoder_contracts import (  # pinned model, fingerprint and pooling constants
    PASSAGE_QUERY_INSTRUCTION, TEXT_ENCODER_DIMENSIONS, TEXT_ENCODER_FILES, TEXT_ENCODER_POOLING, EncoderFingerprint, PinnedModelFiles,  # pinned model, fingerprint and pooling constants
)
from retrieval_encoders.model_file_download import model_folder_for  # where the downloaded files sit
from retrieval_encoders.onnx_session_building import build_inference_session  # CPU session
from retrieval_encoders.word_piece_tokenizing import build_word_piece_tokenizer, pad_encodings_into_inputs  # tokenizing and padding


class PinnedModelTextEncoder:  # implements the TextEncoder protocol of the contract
    def __init__(self, base_folder: Path, files: PinnedModelFiles = TEXT_ENCODER_FILES, thread_count: int = 2, batch_size: int = 32) -> None:  # loads the encoder graph and its tokenizer
        model_folder = model_folder_for(files, base_folder)  # this model's folder
        model_path = model_folder / files.model_file  # the ONNX graph
        self._session = build_inference_session(model_path, thread_count)  # loads the graph once
        self._tokenizer = build_word_piece_tokenizer(model_folder, files, files.max_input_tokens)  # cuts at 512 word pieces
        self._declared_input_names = tuple(model_input.name for model_input in self._session.get_inputs())  # inputs the graph wants
        self._batch_size = batch_size  # texts per graph call
        self._fingerprint = EncoderFingerprint(  # computed once: hashing a 127 MB file takes a moment
            files.model_id, files.revision, compute_file_sha256(model_path), TEXT_ENCODER_DIMENSIONS, TEXT_ENCODER_POOLING, PASSAGE_QUERY_INSTRUCTION,  # model, commit, ONNX checksum, width, pooling, search instruction
        )

    @property  # read-only attribute
    def fingerprint(self) -> EncoderFingerprint:  # which model and files this encoder runs
        return self._fingerprint  # stored in the manifest, compared at service start

    def encode_texts(self, texts: Sequence[str]) -> NDArray[numpy.float32]:  # (len(texts), 384) unit rows, no prefix added here
        vectors = numpy.zeros((len(texts), TEXT_ENCODER_DIMENSIONS), dtype=numpy.float32)  # one row per text, in input order
        if not texts:  # nothing to encode
            return vectors  # empty matrix
        encodings = self._tokenizer.encode_batch(list(texts))  # word pieces, cut at the limit
        order = sorted(range(len(texts)), key=lambda position: len(encodings[position].ids))  # similar lengths together waste less padding
        for batch_start in range(0, len(order), self._batch_size):  # one graph call per batch
            batch_positions = order[batch_start:batch_start + self._batch_size]  # input positions in this batch
            graph_inputs = pad_encodings_into_inputs([encodings[position] for position in batch_positions], self._declared_input_names)  # padded arrays
            hidden_states = self._session.run(None, graph_inputs)[0]  # (batch, tokens, 384) token vectors
            first_token_vectors = hidden_states[:, 0, :].astype(numpy.float32)  # CLS pooling: the first token stands for the text
            lengths = numpy.linalg.norm(first_token_vectors, axis=1, keepdims=True)  # vector lengths
            vectors[batch_positions] = first_token_vectors / numpy.maximum(lengths, 1e-12)  # unit length, never a division by zero
        return vectors  # rows in the caller's order
