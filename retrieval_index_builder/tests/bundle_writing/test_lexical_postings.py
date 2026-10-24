# BM25 postings: weights equal a naive loop on random corpora and rank documents like rank_bm25 does
import math  # naive formula
import random  # seeded corpora
import numpy  # array access
import pytest  # skip when the comparison library is missing
from retrieval_contracts.lexical_tokens import split_lexical_tokens  # the shared word splitter
from retrieval_index_builder.bundle_writing.lexical_postings_writing import build_lexical_postings  # code under test

VOCABULARY = ["stark", "lannister", "wolf", "dragon", "winter", "castle", "sword", "raven", "crown", "river", "fire", "ice", "needle", "harrenhal"]  # invented-enough words


def make_random_corpus(seed: int, document_count: int) -> list[str]:  # documents of random vocabulary words
    random_source = random.Random(seed)  # reproducible
    return [" ".join(random_source.choices(VOCABULARY, k=random_source.randint(3, 30))) for _ in range(document_count)]  # 3 to 30 words each


def naive_weight(documents: list[str], term: str, row: int, k1: float = 1.2, b: float = 0.75) -> float:  # the textbook formula, one document at a time
    tokenized = [split_lexical_tokens(document) for document in documents]  # lexical tokens
    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)  # mean length
    document_frequency = sum(1 for tokens in tokenized if term in tokens)  # documents holding the term
    idf = math.log(1 + (len(tokenized) - document_frequency + 0.5) / (document_frequency + 0.5))  # Lucene idf
    term_count = tokenized[row].count(term)  # occurrences in this document
    return idf * term_count * (k1 + 1) / (term_count + k1 * (1 - b + b * len(tokenized[row]) / average_length))  # BM25 term weight


def read_posting_weight(postings: dict, term: str, row: int) -> float:  # the stored weight of (term, row), 0 when absent
    term_position = int(numpy.searchsorted(postings["terms"], term))  # terms are sorted
    if term_position >= len(postings["terms"]) or postings["terms"][term_position] != term:  # unknown term
        return 0.0  # no postings
    start, end = postings["posting_starts"][term_position], postings["posting_starts"][term_position + 1]  # the term's slice
    rows = postings["posting_rows"][start:end]  # documents holding it
    hits = numpy.nonzero(rows == row)[0]  # position of this row, if present
    return float(postings["posting_weights"][start + hits[0]]) if len(hits) else 0.0  # weight or zero


@pytest.mark.parametrize("seed", [1, 2, 3])  # three random corpora
def test_weights_equal_a_naive_loop(seed):  # every (term, document) weight matches
    documents = make_random_corpus(seed, 25)  # corpus
    postings = build_lexical_postings(documents)  # code under test
    assert list(postings["terms"]) == sorted(postings["terms"])  # terms ascending
    assert postings["posting_starts"][0] == 0 and postings["posting_starts"][-1] == len(postings["posting_rows"])  # slices cover the arrays
    for term in set(VOCABULARY) & set(postings["terms"]):  # words that survived the stopword filter
        for row in range(len(documents)):  # every document
            assert read_posting_weight(postings, term, row) == pytest.approx(naive_weight(documents, term, row), rel=1e-6)  # float32 rounding only


def test_rows_ascend_inside_each_term_and_dtypes_are_fixed():  # layout rules of the bundle format
    postings = build_lexical_postings(make_random_corpus(7, 40))  # corpus
    assert (postings["posting_starts"].dtype, postings["posting_rows"].dtype, postings["posting_weights"].dtype) == (numpy.int64, numpy.int32, numpy.float32)  # types
    for term_position in range(len(postings["terms"])):  # every term
        rows = postings["posting_rows"][postings["posting_starts"][term_position]:postings["posting_starts"][term_position + 1]]  # its rows
        assert list(rows) == sorted(set(rows))  # ascending and distinct


def test_empty_corpus_gives_empty_arrays():  # no documents, no postings
    postings = build_lexical_postings([])  # code under test
    assert len(postings["terms"]) == 0 and list(postings["posting_starts"]) == [0]  # nothing stored


def make_zipf_corpus(seed: int, document_count: int, vocabulary_size: int) -> tuple[list[str], list[str]]:  # a few common words and many rare ones, like real text
    random_source = random.Random(seed)  # reproducible
    words = ["w%d" % number for number in range(vocabulary_size)]  # invented words
    frequencies = [1 / (number + 1) for number in range(vocabulary_size)]  # word number n is n times rarer than word 1
    documents = [" ".join(random_source.choices(words, weights=frequencies, k=random_source.randint(10, 60))) for _ in range(document_count)]  # 10 to 60 words
    return documents, words  # the corpus and its vocabulary


def test_top_three_order_agrees_with_rank_bm25():  # same ranking as the reference library for one-term queries (the two libraries weigh terms against each other differently)
    rank_bm25 = pytest.importorskip("rank_bm25")  # comparison library
    for seed in (1, 2, 3):  # three corpora
        documents, words = make_zipf_corpus(seed, 200, 300)  # corpus
        postings = build_lexical_postings(documents)  # our postings
        reference = rank_bm25.BM25Okapi([split_lexical_tokens(document) for document in documents], k1=1.2, b=0.75)  # reference scorer
        for term in words[30:80]:  # mid-frequency terms
            our_scores = [read_posting_weight(postings, term, row) for row in range(len(documents))]  # our score per document
            reference_scores = reference.get_scores([term])  # their score per document
            our_top = sorted(range(len(documents)), key=lambda row: (-round(our_scores[row], 5), row))[:3]  # our best three, ties by row
            reference_top = sorted(range(len(documents)), key=lambda row: (-round(float(reference_scores[row]), 5), row))[:3]  # their best three
            assert our_top == reference_top  # same three documents in the same order
