# Pure ranking metrics over a list of relevance flags (rank 1 first); no model, no files (DESIGN.md §11.2)
import math  # logarithm of the rank discount
from typing import Sequence  # relevance flag lists


def hit_at_cutoff(relevance: Sequence[bool], rank_cutoff: int) -> float:  # 1.0 when any of the first rank_cutoff results is relevant
    return 1.0 if any(relevance[:rank_cutoff]) else 0.0  # binary hit


def precision_at_cutoff(relevance: Sequence[bool], rank_cutoff: int) -> float:  # share of the first rank_cutoff results that are relevant
    return sum(relevance[:rank_cutoff]) / rank_cutoff  # a short list still divides by the cutoff (missing results are misses)


def recall_at_cutoff(relevance: Sequence[bool], rank_cutoff: int, relevant_total: int) -> float:  # relevant found over relevant in the index
    if relevant_total == 0:  # nothing to find
        return 0.0  # such a question is dropped by the callers, never rewarded here
    return sum(relevance[:rank_cutoff]) / relevant_total  # found share


def reciprocal_rank_at_cutoff(relevance: Sequence[bool], rank_cutoff: int) -> float:  # 1 / rank of the first relevant result, 0 when none
    for rank, is_relevant in enumerate(relevance[:rank_cutoff], start=1):  # ranks are 1-based
        if is_relevant:  # first relevant result
            return 1.0 / rank  # its reciprocal rank
    return 0.0  # no relevant result inside the cutoff


def normalized_discounted_cumulative_gain(relevance: Sequence[bool], rank_cutoff: int, relevant_total: int) -> float:
    """Binary gain, discount 1 / log2(rank + 1); the ideal list holds min(relevant_total, rank_cutoff) relevant results first."""
    gain = sum(1.0 / math.log2(rank + 1) for rank, is_relevant in enumerate(relevance[:rank_cutoff], start=1) if is_relevant)  # actual gain
    ideal_gain = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(relevant_total, rank_cutoff) + 1))  # best possible gain
    return gain / ideal_gain if ideal_gain > 0 else 0.0  # no relevant result exists: nothing to normalise


def take_packed_passages(token_counts: Sequence[int], budget_tokens: int, max_passages: int) -> int:  # how many passages the packer takes whole
    taken, used_tokens = 0, 0  # passages taken and tokens they use
    for token_count in token_counts:  # in rank order
        if taken == max_passages or used_tokens + token_count > budget_tokens:  # limit reached or the next one does not fit
            break  # packing stops at the first passage that does not fit
        taken, used_tokens = taken + 1, used_tokens + token_count  # take it
    return taken  # count of whole passages inside the budget


def hit_within_packed_budget(relevance: Sequence[bool], token_counts: Sequence[int], budget_tokens: int, max_passages: int) -> float:
    """1.0 when a relevant passage is among those taken in rank order until the budget is used (the chunk-size comparison measure)."""
    taken = take_packed_passages(token_counts, budget_tokens, max_passages)  # passages that fit
    return 1.0 if any(relevance[:taken]) else 0.0  # binary hit among them


def context_precision(relevance: Sequence[bool], token_counts: Sequence[int], budget_tokens: int, max_passages: int) -> float | None:
    """Share of the PACKED passages that are relevant; None when nothing fits."""
    taken = take_packed_passages(token_counts, budget_tokens, max_passages)  # passages that fit
    return sum(relevance[:taken]) / taken if taken else None  # relevant share of the packed ones


def average_values(values: Sequence[float]) -> float:  # arithmetic mean of per-question values (0.0 for an empty list)
    return sum(values) / len(values) if values else 0.0  # mean


def percentile_value(values: Sequence[float], percentile: float) -> float:  # nearest-rank percentile (95.0 = p95); 0.0 for an empty list
    if not values:  # nothing measured
        return 0.0  # no latency to report
    ordered_values = sorted(values)  # ascending
    rank = max(1, math.ceil(round(percentile / 100.0 * len(ordered_values), 9)))  # nearest rank, 1-based
    return ordered_values[rank - 1]  # the value at that rank
