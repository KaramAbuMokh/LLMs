# Which passages count as holding the answer of a held-out question: those containing one of its gold hits WHOLE (no widening, DESIGN.md §11.2)
from typing import Sequence  # lists of records and hits
from retrieval_contracts.bundle_records import PassageRecord  # a stored passage
from retrieval_contracts.rare_shingle_location import ShingleHit  # a gold hit: a character span of one part text


def contains_hit_whole(record: PassageRecord, hit: ShingleHit) -> bool:  # True when the hit lies wholly inside the passage
    return (record.part_number == hit.part_number and record.character_start <= hit.character_start  # same part, starts inside
            and hit.character_end <= record.character_end)  # and ends inside (a hit cut by the passage edge is not contained)


def is_passage_relevant(record: PassageRecord, gold_hits: Sequence[ShingleHit]) -> bool:  # True when any gold hit sits whole inside the passage
    return any(contains_hit_whole(record, hit) for hit in gold_hits)  # one whole hit is enough


def flag_relevant_passages(ranked_records: Sequence[PassageRecord], gold_hits: Sequence[ShingleHit]) -> list[bool]:  # relevance flags, rank 1 first
    return [is_passage_relevant(record, gold_hits) for record in ranked_records]  # one flag per ranked passage


def count_relevant_passages(index_records: Sequence[PassageRecord], gold_hits: Sequence[ShingleHit]) -> int:  # relevant passages in the whole index
    return sum(1 for record in index_records if is_passage_relevant(record, gold_hits))  # recall and ideal gain read this
