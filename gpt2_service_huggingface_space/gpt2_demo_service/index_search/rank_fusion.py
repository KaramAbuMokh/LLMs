# Reciprocal rank fusion: merges several ranked id lists into one, rewarding items that rank well in any list
from dataclasses import dataclass  # frozen result record
from typing import Sequence  # input lists


@dataclass(frozen=True)
class FusedCandidate:  # one item of the fused ranking
    item_id: str  # the passage or pair id
    fused_score: float  # sum of 1 / (constant + rank) over the lists that hold it
    ranks: tuple[int | None, ...]  # 1-based rank in each input list, None where the list lacks the item


def fuse_rankings(ranked_id_lists: Sequence[Sequence[str]], fusion_constant: int) -> list[FusedCandidate]:  # best first, ties by id
    item_ids = sorted({item_id for ranked_ids in ranked_id_lists for item_id in ranked_ids})  # every item once, in id order
    rank_tables = [{item_id: rank for rank, item_id in enumerate(ranked_ids, start=1)} for ranked_ids in ranked_id_lists]  # id -> 1-based rank per list
    candidates = []  # one entry per item
    for item_id in item_ids:  # every item
        ranks = tuple(rank_table.get(item_id) for rank_table in rank_tables)  # its rank in each list
        fused_score = sum(1.0 / (fusion_constant + rank) for rank in ranks if rank is not None)  # reciprocal rank sum
        candidates.append(FusedCandidate(item_id, fused_score, ranks))  # keep it
    return sorted(candidates, key=lambda candidate: -candidate.fused_score)  # stable sort keeps id order for equal scores
