# Deterministic stand-ins for the two encoders, so every package can test without model files (tests only, never loaded by the service)
import hashlib  # token hashing that is the same in every process (Python's own hash() is randomised)
from typing import Sequence  # input type
import numpy  # vectors
from retrieval_contracts.encoder_contracts import EncoderFingerprint  # what the fixture bundle's manifest records
from retrieval_contracts.lexical_tokens import split_lexical_tokens  # word splitting shared with lexical search

HASHING_ENCODER_DIMENSIONS = 256  # wide enough that unrelated words rarely share a column, still tiny
HASHING_ENCODER_FINGERPRINT = EncoderFingerprint(  # fingerprint written into the fixture bundle's manifest
    "fixture/hashing-text-encoder", "fixture-revision-1", "0" * 64, HASHING_ENCODER_DIMENSIONS, "token-hashing", "search passages: ",
)


def hash_token_to_column(token: str) -> int:  # the vector column one lexical token adds to
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big") % HASHING_ENCODER_DIMENSIONS  # stable column


class HashingTextEncoder:  # bag of lexical tokens hashed into 256 columns, scaled to unit length; texts sharing words score high
    @property
    def fingerprint(self) -> EncoderFingerprint:  # satisfies the TextEncoder protocol
        return HASHING_ENCODER_FINGERPRINT  # fixed fixture fingerprint

    def encode_texts(self, texts: Sequence[str]) -> numpy.ndarray:  # (len(texts), 256) float32 rows, unit length (all-zero for a text with no tokens)
        vectors = numpy.zeros((len(texts), HASHING_ENCODER_DIMENSIONS), dtype=numpy.float32)  # one row per text
        for row, text in enumerate(texts):  # every text
            for token in split_lexical_tokens(text):  # every content word, repeats counted
                vectors[row, hash_token_to_column(token)] += 1.0  # add to its column
        lengths = numpy.linalg.norm(vectors, axis=1, keepdims=True)  # row lengths
        return numpy.divide(vectors, lengths, out=numpy.zeros_like(vectors), where=lengths > 0)  # unit rows, empty rows stay zero


class TokenOverlapPairScorer:  # stand-in reranker: the number of distinct lexical tokens a passage shares with the question
    def score_pairs(self, question: str, passage_texts: Sequence[str]) -> numpy.ndarray:  # one float32 score per passage
        question_tokens = set(split_lexical_tokens(question))  # distinct question words
        overlaps = [len(question_tokens & set(split_lexical_tokens(text))) for text in passage_texts]  # shared words per passage
        return numpy.array(overlaps, dtype=numpy.float32)  # scores in passage order
