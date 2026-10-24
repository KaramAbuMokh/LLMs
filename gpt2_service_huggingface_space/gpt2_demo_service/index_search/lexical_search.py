# Lexical search over precomputed BM25 postings: the score of a document is the sum of the weights of the distinct query words it holds
import numpy  # arrays
from numpy.typing import NDArray  # array type hints
from gpt2_demo_service.index_loading.bundle_loading import LexicalPostings  # the loaded postings
from gpt2_demo_service.index_search.evidence_records import RankedRow  # one ranked row
from retrieval_contracts.lexical_tokens import split_lexical_tokens  # the same word splitter the index was built with


def score_lexical_matches(postings: LexicalPostings, text: str) -> NDArray[numpy.float32]:  # (document_count,) scores, 0 for documents without a query word
    term_rows = [postings.term_rows[token] for token in set(split_lexical_tokens(text)) if token in postings.term_rows]  # distinct known words
    if not term_rows:  # no query word is in the index
        return numpy.zeros(postings.document_count, dtype=numpy.float32)  # nothing matches
    spans = [(postings.posting_starts[row], postings.posting_starts[row + 1]) for row in term_rows]  # each word's postings
    document_rows = numpy.concatenate([postings.posting_rows[start:end] for start, end in spans])  # documents holding the words
    weights = numpy.concatenate([postings.posting_weights[start:end] for start, end in spans])  # their BM25 weights
    summed = numpy.bincount(document_rows, weights=weights, minlength=postings.document_count)  # sum per document
    return summed.astype(numpy.float32)  # same type as the dense scores


def find_top_lexical_matches(postings: LexicalPostings, text: str, limit: int) -> list[RankedRow]:  # score > 0, best first, ties by row
    scores = score_lexical_matches(postings, text)  # every document's score
    matching_rows = numpy.flatnonzero(scores > 0)  # only documents with at least one query word
    order = numpy.lexsort((matching_rows, -scores[matching_rows]))  # score descending, then row ascending
    return [RankedRow(int(matching_rows[position]), float(scores[matching_rows[position]])) for position in order[:max(limit, 0)]]  # the top rows
