# Tests of reciprocal rank fusion: the fixture example, ties by id, ranks per list
from gpt2_demo_service.index_search.rank_fusion import fuse_rankings  # code under test


def test_fixture_example(fixture_expectations):  # the recorded example with k = 60
    example = fixture_expectations["rank_fusion_example"]  # lists and expected fused scores
    fused = fuse_rankings(example["lists"], example["constant"])  # our result
    assert [candidate.item_id for candidate in fused] == [item[0] for item in example["fused"]]  # same order
    assert all(abs(candidate.fused_score - item[1]) < 1e-8 for candidate, item in zip(fused, example["fused"]))  # same scores


def test_ranks_are_one_based_and_none_where_missing():  # each candidate reports its rank per list
    fused = {candidate.item_id: candidate for candidate in fuse_rankings([["a", "b"], ["b", "c"]], 60)}  # fuse two lists
    assert fused["a"].ranks == (1, None) and fused["b"].ranks == (2, 1) and fused["c"].ranks == (None, 2)  # per-list ranks


def test_equal_scores_are_ordered_by_id():  # deterministic ties
    fused = fuse_rankings([["zebra"], ["apple"]], 60)  # both rank first in their list
    assert [candidate.item_id for candidate in fused] == ["apple", "zebra"]  # id ascending


def test_empty_input_gives_empty_output():  # no lists, no candidates
    assert fuse_rankings([], 60) == [] and fuse_rankings([[], []], 60) == []  # nothing to fuse
