# Exact dense search: inner product of unit vectors, so the score is the cosine; no approximation
import numpy  # arrays
from numpy.typing import NDArray  # array type hints
from gpt2_demo_service.index_search.evidence_records import RankedRow  # one ranked row


def find_top_dense_matches(vectors: NDArray[numpy.float32], query_vector: NDArray[numpy.float32], limit: int) -> list[RankedRow]:  # best first, ties by row
    if limit <= 0 or len(vectors) == 0:  # nothing requested or nothing indexed
        return []  # empty ranking
    scores = vectors @ query_vector  # cosine of every row
    if limit >= len(scores):  # everything is wanted
        candidate_rows = numpy.arange(len(scores))  # all rows
    else:  # keep every row at or above the limit-th best score, so ties at the border are not cut arbitrarily
        border_score = numpy.partition(scores, len(scores) - limit)[len(scores) - limit]  # the limit-th best score
        candidate_rows = numpy.flatnonzero(scores >= border_score)  # rows that may enter the top
    order = numpy.lexsort((candidate_rows, -scores[candidate_rows]))  # score descending, then row ascending
    return [RankedRow(int(candidate_rows[position]), float(scores[candidate_rows[position]])) for position in order[:limit]]  # the top rows
