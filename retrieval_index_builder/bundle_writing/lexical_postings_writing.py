# Builds the lexical postings (BM25 weights precomputed per term and document) stored in the bundle's .npz files (DESIGN.md 6.3, D6)
import math  # natural logarithm for the inverse document frequency
from collections import Counter  # term counts per document
from typing import Sequence  # documents in
import numpy  # stored arrays
from numpy.typing import NDArray  # result type
from retrieval_contracts.lexical_tokens import split_lexical_tokens  # the one word splitter shared with the service


def build_lexical_postings(documents: Sequence[str], saturation_k1: float = 1.2, length_normalization_b: float = 0.75) -> dict[str, NDArray]:  # postings of one document list, with the BM25 constants of the design
    term_postings: dict[str, list[tuple[int, int]]] = {}  # term -> (row, count) in ascending row order
    document_lengths = []  # tokens per document, stopwords already dropped
    for row, document in enumerate(documents):  # documents in row order
        tokens = split_lexical_tokens(document)  # lexical tokens of the document
        document_lengths.append(len(tokens))  # length for the normalisation
        for term, term_count in Counter(tokens).items():  # each distinct term once
            term_postings.setdefault(term, []).append((row, term_count))  # remember where it occurs
    document_count = len(documents)  # N in the inverse document frequency
    average_length = (sum(document_lengths) / document_count) if document_count else 1.0  # mean document length
    terms = sorted(term_postings)  # ascending order, as the service binary-searches nothing but expects sorted terms
    posting_starts, posting_rows, posting_weights = [0], [], []  # flat arrays; term i owns [starts[i], starts[i + 1])
    for term in terms:  # every term
        postings = term_postings[term]  # its rows
        inverse_document_frequency = math.log(1 + (document_count - len(postings) + 0.5) / (len(postings) + 0.5))  # Lucene idf
        for row, term_count in postings:  # every document holding the term
            length_factor = 1 - length_normalization_b + length_normalization_b * document_lengths[row] / average_length  # long documents count less
            posting_rows.append(row)  # the document row
            posting_weights.append(inverse_document_frequency * term_count * (saturation_k1 + 1) / (term_count + saturation_k1 * length_factor))  # full term weight
        posting_starts.append(len(posting_rows))  # where the next term starts
    return {  # the four arrays of the bundle format
        "terms": numpy.array(terms, dtype="<U1") if not terms else numpy.array(terms),  # unicode, sorted ascending
        "posting_starts": numpy.array(posting_starts, dtype=numpy.int64),  # terms + 1 entries
        "posting_rows": numpy.array(posting_rows, dtype=numpy.int32),  # ascending within a term
        "posting_weights": numpy.array(posting_weights, dtype=numpy.float32),  # the full BM25 term weight
    }
