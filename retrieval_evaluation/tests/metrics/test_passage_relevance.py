# A passage is relevant when it holds a gold hit WHOLE; one that cuts a hit is not (DESIGN.md §11.2)
from retrieval_contracts.bundle_records import PassageRecord  # a stored passage
from retrieval_contracts.rare_shingle_location import ShingleHit  # a gold hit
from retrieval_evaluation.metrics.passage_relevance import count_relevant_passages, flag_relevant_passages, is_passage_relevant  # under test


def make_passage(part_number: int, start: int, end: int) -> PassageRecord:  # a passage record with only the span that matters
    return PassageRecord("bp-%d-000-%d" % (part_number, start), part_number, "Book", 1, 1, "Name", start, end, 10, "x" * (end - start))  # invented


def test_whole_hit_is_relevant_and_cut_hit_is_not():  # containment, not overlap
    hit = ShingleHit(1, 100, 130)  # gold hit
    assert is_passage_relevant(make_passage(1, 90, 140), [hit])  # holds it whole
    assert is_passage_relevant(make_passage(1, 100, 130), [hit])  # edges count as inside
    assert not is_passage_relevant(make_passage(1, 90, 120), [hit])  # cuts the end
    assert not is_passage_relevant(make_passage(1, 110, 200), [hit])  # cuts the start
    assert not is_passage_relevant(make_passage(2, 90, 140), [hit])  # another part


def test_any_gold_hit_is_enough_and_counts_cover_the_index():  # flags and totals
    hits = [ShingleHit(1, 100, 130), ShingleHit(1, 500, 530)]  # two gold hits
    ranked = [make_passage(1, 0, 90), make_passage(1, 480, 600), make_passage(1, 95, 135)]  # three passages
    assert flag_relevant_passages(ranked, hits) == [False, True, True]  # rank 2 holds the second hit, rank 3 the first
    assert count_relevant_passages(ranked, hits) == 2  # relevant passages in the index
