# Hand cases of the ranking measures (DESIGN.md §14.3): relevance [0,1,0,1,0] with two relevant passages in the index
import pytest  # approximate comparisons
from retrieval_evaluation.metrics import ranking_metrics  # measures under test

RELEVANCE = [False, True, False, True, False]  # ranks 2 and 4 are relevant


def test_hit_precision_and_recall():  # the cutoff measures
    assert ranking_metrics.hit_at_cutoff(RELEVANCE, 1) == 0.0  # rank 1 is not relevant
    assert ranking_metrics.hit_at_cutoff(RELEVANCE, 3) == 1.0  # rank 2 is
    assert ranking_metrics.precision_at_cutoff(RELEVANCE, 3) == pytest.approx(1 / 3)  # one of three
    assert ranking_metrics.recall_at_cutoff(RELEVANCE, 3, 2) == pytest.approx(1 / 2)  # one of two relevant found
    assert ranking_metrics.recall_at_cutoff(RELEVANCE, 5, 2) == 1.0  # both found


def test_reciprocal_rank():  # first relevant at rank 2
    assert ranking_metrics.reciprocal_rank_at_cutoff(RELEVANCE, 10) == 0.5  # 1 / 2
    assert ranking_metrics.reciprocal_rank_at_cutoff([False, False], 10) == 0.0  # none relevant


def test_normalized_discounted_cumulative_gain():  # the two numbers of the design
    assert ranking_metrics.normalized_discounted_cumulative_gain(RELEVANCE, 5, 2) == pytest.approx(0.650921, abs=1e-6)  # at 5
    assert ranking_metrics.normalized_discounted_cumulative_gain(RELEVANCE, 3, 2) == pytest.approx(0.386853, abs=1e-6)  # at 3
    assert ranking_metrics.normalized_discounted_cumulative_gain([False], 5, 0) == 0.0  # nothing relevant exists


def test_hit_within_packed_budget():  # whole passages in rank order until the budget or the passage limit
    token_counts = [200, 200, 150, 100, 100]  # prompt cost per rank
    assert ranking_metrics.take_packed_passages(token_counts, 450, 3) == 2  # 200 + 200 fit, 150 more does not
    assert ranking_metrics.take_packed_passages(token_counts, 1000, 3) == 3  # the passage limit stops it
    assert ranking_metrics.hit_within_packed_budget(RELEVANCE, token_counts, 450, 3) == 1.0  # rank 2 is inside
    assert ranking_metrics.hit_within_packed_budget(RELEVANCE, token_counts, 150, 3) == 0.0  # nothing fits, so no hit
    assert ranking_metrics.hit_within_packed_budget([False, False, True], token_counts, 450, 3) == 0.0  # relevant passage is packed out


def test_context_precision():  # share of packed passages that are relevant
    assert ranking_metrics.context_precision(RELEVANCE, [100] * 5, 300, 3) == pytest.approx(1 / 3)  # one of three packed
    assert ranking_metrics.context_precision(RELEVANCE, [500] * 5, 300, 3) is None  # nothing fits


def test_percentile_and_average():  # nearest-rank percentile and mean
    assert ranking_metrics.percentile_value(list(range(1, 21)), 95.0) == 19  # 19th of 20
    assert ranking_metrics.percentile_value([], 95.0) == 0.0  # empty
    assert ranking_metrics.average_values([1.0, 2.0, 3.0]) == 2.0  # mean
