# Decides which book part and chapter a question-answer pair came from (DESIGN.md 8.5), using the shared rare-shingle locator
from dataclasses import dataclass  # frozen result
from typing import Literal, Mapping, Sequence  # method names and inputs
from retrieval_contracts.book_parts import locate_cleaned_offset  # coarse book_location -> (part, offset)
from retrieval_contracts.bundle_records import ChapterRecord  # chapter spans
from retrieval_contracts.rare_shingle_location import LocatedSpan, RareShingleIndex, SearchWindow, locate_text  # the one answer locator
from retrieval_index_builder.source_alignment.chapter_alignment import find_chapter_at  # chapter holding one character

PlacementMethod = Literal["shingles_in_window", "shingles_corpus_wide", "window_inside_chapter", "window_only", "unplaced"]  # how a pair was placed


@dataclass(frozen=True)  # immutable record
class PairPlacement:  # where one pair sits in the books
    part_number: int | None  # book part, None when unplaced
    chapter: ChapterRecord | None  # chapter, None when unknown (appendix, unaligned book, wide window)
    method: PlacementMethod  # which rule placed it


def build_search_window(book_location: Mapping[str, int] | None, part_lengths: Mapping[int, int], window_characters: int) -> SearchWindow | None:  # the coarse location of a pair as a clamped search window
    if not book_location:  # the pair was never located
        return None  # no window
    mapped_location = locate_cleaned_offset(book_location["book"], book_location["offset"], part_lengths)  # (part, offset) or None for trimmed matter
    if mapped_location is None:  # points at text the cleaning removed
        return None  # treated as unlocated
    part_number, offset_in_part = mapped_location  # the part and the place inside it
    return SearchWindow(part_number, max(0, offset_in_part - window_characters), min(part_lengths[part_number], offset_in_part + window_characters))  # clamped


def find_chapter_of_span(chapters: Sequence[ChapterRecord], located_span: LocatedSpan) -> ChapterRecord | None:  # the chapter holding the span's centre
    return find_chapter_at(chapters, located_span.part_number, (located_span.character_start + located_span.character_end) // 2)  # centre character


def place_by_window(search_window: SearchWindow, chapters: Sequence[ChapterRecord]) -> PairPlacement:  # no shingle cluster: use only what the window says
    inside_chapter = next(  # a chapter that holds the whole window
        (chapter for chapter in chapters if chapter.part_number == search_window.part_number  # chapters of the window's part
         and chapter.character_start <= search_window.character_start and search_window.character_end <= chapter.character_end), None,  # that contain the whole window
    )
    if inside_chapter is not None:  # the 40,000-character window sits inside one chapter (rare)
        return PairPlacement(search_window.part_number, inside_chapter, "window_inside_chapter")  # book and chapter
    return PairPlacement(search_window.part_number, None, "window_only")  # book only


def place_pair(question: str, answer: str, book_location: Mapping[str, int] | None, part_lengths: Mapping[int, int], chapters: Sequence[ChapterRecord],  # book part and chapter of one pair
               shingle_index: RareShingleIndex, window_characters: int) -> PairPlacement:  # part and chapter of one pair
    searched_text = question + " " + answer  # names in either side count
    search_window = build_search_window(book_location, part_lengths, window_characters)  # the coarse stretch, if the pair has one
    if search_window is not None:  # a coarse location exists: search only inside it, so it is never contradicted
        located_span = locate_text(shingle_index, searched_text, search_window)  # rare shingles inside the window
        if located_span is None:  # no cluster
            return place_by_window(search_window, chapters)  # window-only placement
        return PairPlacement(located_span.part_number, find_chapter_of_span(chapters, located_span), "shingles_in_window")  # placed by the text
    located_span = locate_text(shingle_index, searched_text)  # no window: search every part
    if located_span is None:  # nothing found anywhere
        return PairPlacement(None, None, "unplaced")  # book and chapter unknown
    return PairPlacement(located_span.part_number, find_chapter_of_span(chapters, located_span), "shingles_corpus_wide")  # placed by the text
