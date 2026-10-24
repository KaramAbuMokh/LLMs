# Finds the book passage that holds the answer of a training pair: the pair's rare shingle hits fall inside it (frozen locator, DESIGN.md section 6.6)
from typing import Mapping, Sequence  # inputs
from retrieval_contracts.bundle_records import PassageRecord  # stored passages
from retrieval_contracts.rare_shingle_location import LocatedSpan, RareShingleIndex, SearchWindow, locate_text  # the shared answer locator
from retrieval_index_builder.source_alignment.pair_placement import build_search_window  # coarse location -> clamped window


def locate_pair_span(question: str, answer: str, book_location: Mapping[str, int] | None, part_lengths: Mapping[int, int],
                     shingle_index: RareShingleIndex, window_characters: int) -> LocatedSpan | None:  # where the pair's text sits in the books, as the index builder placed it
    search_window: SearchWindow | None = build_search_window(book_location, part_lengths, window_characters)  # the coarse stretch, if any
    return locate_text(shingle_index, question + " " + answer, search_window)  # inside the window, else corpus-wide


def count_hits_inside(passage: PassageRecord, span: LocatedSpan) -> int:  # how many of the span's hits the passage holds whole
    return sum(hit.part_number == passage.part_number and passage.character_start <= hit.character_start and hit.character_end <= passage.character_end
               for hit in span.hits)  # a hit counts only when it is wholly inside


def find_oracle_passage(span: LocatedSpan | None, passages_by_part: Mapping[int, Sequence[PassageRecord]], minimum_hits: int, excluded_part: int) -> PassageRecord | None:  # the passage with the most hits, or None
    if span is None or span.part_number == excluded_part:  # unlocated, or inside the evaluation's own stretch
        return None  # no oracle
    best_passage, best_count = None, 0  # best so far
    for passage in passages_by_part.get(span.part_number, ()):  # only passages of the span's part
        if passage.character_end <= span.character_start or passage.character_start >= span.character_end:  # cannot hold a hit
            continue  # next passage
        hit_count = count_hits_inside(passage, span)  # hits inside this passage
        if hit_count > best_count:  # strictly more than the best so far (ties keep the earlier passage)
            best_passage, best_count = passage, hit_count  # new best
    return best_passage if best_count >= minimum_hits else None  # too few hits is no proof
