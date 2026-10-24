# Encodes texts in batches and stores the vectors as plain float32 .npy files (no pickles)
from pathlib import Path  # output file
from typing import Sequence  # texts in
import numpy  # vectors
from numpy.typing import NDArray  # result type
from retrieval_contracts.encoder_contracts import TextEncoder  # the shared encoder interface


def encode_in_batches(encoder: TextEncoder, texts: Sequence[str], batch_size: int) -> NDArray[numpy.float32]:  # (len(texts), dimensions) unit rows
    batches = [encoder.encode_texts(texts[start:start + batch_size]) for start in range(0, len(texts), batch_size)]  # one encoder call per batch
    if not batches:  # no texts at all
        return numpy.zeros((0, 0), dtype=numpy.float32)  # empty matrix
    return numpy.concatenate(batches).astype(numpy.float32)  # rows in text order


def save_vectors(vector_path: Path, vectors: NDArray[numpy.float32]) -> None:  # one vector file of the bundle
    numpy.save(vector_path, vectors, allow_pickle=False)  # raw float32 matrix, row i belongs to record line i


def save_lexical_postings(postings_path: Path, postings: dict[str, NDArray]) -> None:  # one lexical file of the bundle
    numpy.savez_compressed(postings_path, **postings)  # four named arrays, no pickles
