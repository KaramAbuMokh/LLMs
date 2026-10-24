# Tests of the oracle passage rule: the passage holding the most of the pair's rare hits, at least two, never in the evaluation's own part
from oracle_passage_finding import count_hits_inside, find_oracle_passage  # code under test
from retrieval_contracts.bundle_records import PassageRecord  # stored passage record
from retrieval_contracts.rare_shingle_location import LocatedSpan, ShingleHit  # locator results

MINIMUM_HITS = 2  # the production value
EXCLUDED_PART = 6  # the evaluation's own stretch


def make_passage(passage_id: str, part_number: int, start: int, end: int) -> PassageRecord:  # a passage with invented text
    return PassageRecord(passage_id, part_number, "Book", 1, 1, "Title", start, end, 20, "invented text")  # only the span matters here


def make_span(part_number: int, hit_starts: list[int]) -> LocatedSpan:  # a located span with 20-character hits at the given starts
    hits = tuple(ShingleHit(part_number, start, start + 20) for start in hit_starts)  # one hit per start
    return LocatedSpan(part_number, hits[0].character_start, hits[-1].character_end, hits)  # the cluster


def test_passage_with_most_hits_wins() -> None:  # two passages overlap the span; the one with three hits is the oracle
    passages = {0: [make_passage("a", 0, 0, 100), make_passage("b", 0, 100, 300)]}  # neighbours
    span = make_span(0, [90, 110, 150, 200])  # one hit straddles nothing: 90-110 lies in neither, three lie in b
    assert find_oracle_passage(span, passages, MINIMUM_HITS, EXCLUDED_PART).passage_id == "b"  # b holds 110, 150, 200


def test_one_hit_is_not_proof() -> None:  # a passage with a single hit is not an oracle
    passages = {0: [make_passage("a", 0, 0, 200)]}  # one passage
    assert find_oracle_passage(make_span(0, [10, 500]), passages, MINIMUM_HITS, EXCLUDED_PART) is None  # only one hit inside


def test_held_out_part_and_missing_span_give_none() -> None:  # the evaluation's text is never an oracle
    passages = {6: [make_passage("held-out", 6, 0, 200)]}  # a passage of part 6
    assert find_oracle_passage(make_span(6, [10, 40]), passages, MINIMUM_HITS, EXCLUDED_PART) is None  # excluded part
    assert find_oracle_passage(None, passages, MINIMUM_HITS, EXCLUDED_PART) is None  # unlocated pair


def test_ties_keep_the_earlier_passage() -> None:  # equal counts: the first passage stays
    passages = {0: [make_passage("a", 0, 0, 100), make_passage("b", 0, 100, 200)]}  # two passages
    assert find_oracle_passage(make_span(0, [10, 40, 110, 140]), passages, MINIMUM_HITS, EXCLUDED_PART).passage_id == "a"  # two hits each


def test_hits_must_lie_wholly_inside() -> None:  # a hit that crosses the passage edge does not count
    assert count_hits_inside(make_passage("a", 0, 0, 100), make_span(0, [90, 50])) == 1  # 90-110 crosses the edge, 50-70 is inside
