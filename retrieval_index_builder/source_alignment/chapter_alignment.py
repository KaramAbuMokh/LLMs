# Finds every expected chapter heading in the cleaned part texts and cuts each part into chapter spans (the hard part of the build, DESIGN.md 8.3)
import re  # whitespace after a heading
from dataclasses import dataclass  # frozen results
from typing import Mapping, Sequence  # inputs
from retrieval_contracts.book_parts import BOOK_PARTS, BookPart  # which parts make a book, in reading order
from retrieval_contracts.bundle_records import ChapterRecord  # the stored chapter type
from retrieval_index_builder.source_alignment.heading_patterns import build_heading_pattern  # heading patterns

MINIMUM_CHAPTER_CHARACTERS = 3000  # a heading is accepted at least this far after the previous heading's start
LEADING_WHITESPACE_PATTERN = re.compile(r"\s*")  # spaces between a heading and the chapter text
APPENDIX_START_PATTERN = re.compile(r"A\s?PPENDIX|WESTEROS THE BOY KING")  # the houses-and-characters appendix that follows the last chapter of a book


@dataclass(frozen=True)  # immutable record
class MatchedHeading:  # one expected heading found in a book's combined text
    chapter_number: int  # 0 = prologue, then reading order
    chapter_title: str  # "Tyrion", "Prologue", "The Lost Lord"
    heading_start: int  # where the heading begins in the combined text (the previous chapter ends here)
    content_start: int  # first character of the chapter text after the heading


@dataclass(frozen=True)  # immutable record
class ChapterAlignment:  # chapters of every book that passed the gate, and what happened per book
    chapters: tuple[ChapterRecord, ...]  # all kept chapter records, parts in part-number order
    report: Mapping[str, object]  # per book: expected, found, unmatched headings, shortest and longest chapter, gate result


def find_chapter_at(chapters: Sequence[ChapterRecord], part_number: int, offset: int) -> ChapterRecord | None:  # the chapter holding one character
    return next((chapter for chapter in chapters if chapter.part_number == part_number and chapter.character_start <= offset < chapter.character_end), None)  # headings belong to none


def match_headings(combined_text: str, headings: Sequence[str]) -> list[MatchedHeading | None]:  # one entry per expected heading, None when not found
    matches: list[MatchedHeading | None] = []  # results in heading order
    previous_start: int | None = None  # start of the last accepted heading
    for chapter_number, heading in enumerate(headings):  # reading order
        heading_pattern = build_heading_pattern(heading)  # how to find it
        search_from = 0 if previous_start is None else previous_start + MINIMUM_CHAPTER_CHARACTERS  # forward only, never inside the last chapter's opening
        match = heading_pattern.pattern.search(combined_text, search_from)  # first hit
        if match and chapter_number + 1 < len(headings):  # is the next heading already in sight before this hit?
            next_match = build_heading_pattern(headings[chapter_number + 1]).pattern.search(combined_text, search_from)  # first hit of the next one
            if next_match and next_match.start() < match.start():  # then this heading is missing from the cleaned text
                match = None  # leave it unmatched instead of jumping over the next chapter
        if match is None:  # not found
            matches.append(None)  # reported as unmatched; its text joins the previous chapter
            continue  # try the next heading from the same place
        title = heading_pattern.title or match.group("name").capitalize()  # known title, or the name read from the text
        content_start = LEADING_WHITESPACE_PATTERN.match(combined_text, match.end()).end()  # chapter text starts after the heading
        matches.append(MatchedHeading(chapter_number, title, match.start(), content_start))  # accepted
        previous_start = match.start()  # the next heading must come later
    return matches  # aligned and missing headings


def find_story_end(combined_text: str, last_heading: MatchedHeading) -> int:  # where the last chapter stops and the appendix (no chapter) begins
    appendix_match = APPENDIX_START_PATTERN.search(combined_text, last_heading.content_start)  # the appendix title inside the last chapter's text
    return appendix_match.start() if appendix_match else len(combined_text)  # the end of the book when there is no appendix


def cut_into_part_records(matched: Sequence[MatchedHeading], book_parts: Sequence[BookPart], part_lengths: Mapping[int, int], story_end: int) -> list[ChapterRecord]:  # chapter spans per part, a chapter crossing a part edge gets one piece in each
    part_starts, running_start = {}, 0  # where each part begins in the combined text
    for book_part in book_parts:  # parts in reading order
        part_starts[book_part.part_number] = running_start  # offset of its first character
        running_start += part_lengths[book_part.part_number]  # the next part follows directly
    records = []  # chapter pieces, one per chapter and part
    for position, heading in enumerate(matched):  # every matched heading
        chapter_end = matched[position + 1].heading_start if position + 1 < len(matched) else story_end  # next heading, or where the appendix begins
        for book_part in book_parts:  # a chapter may continue into the next part (Feast parts 3, 6, 4)
            part_start = part_starts[book_part.part_number]  # this part's first offset in the combined text
            piece_start = max(heading.content_start, part_start)  # the piece cannot begin before the chapter text or the part
            piece_end = min(chapter_end, part_start + part_lengths[book_part.part_number])  # nor end after the chapter or the part
            if piece_start < piece_end:  # the chapter reaches into this part
                records.append(ChapterRecord(book_part.part_number, heading.chapter_number, heading.chapter_title, piece_start - part_start, piece_end - part_start))  # the piece of this chapter inside this part
    return records  # unsorted by part: the caller sorts


def align_book(book_index: int, part_texts: Mapping[int, str], headings: Sequence[str], minimum_aligned_share: float) -> tuple[list[ChapterRecord], dict]:  # aligns one book and reports on it
    book_parts = sorted((part for part in BOOK_PARTS if part.location_book_index == book_index), key=lambda part: part.reading_order)  # parts in reading order
    part_lengths = {part.part_number: len(part_texts[part.part_number]) for part in book_parts}  # characters per part
    combined_text = "".join(part_texts[part.part_number] for part in book_parts)  # the book as one text, parts in reading order
    matches = match_headings(combined_text, headings)  # headings found in it
    matched = [match for match in matches if match is not None]  # the ones found
    story_end = find_story_end(combined_text, matched[-1]) if matched else 0  # chapters stop where the appendix begins
    records = cut_into_part_records(matched, book_parts, part_lengths, story_end)  # chapter spans per part
    aligned_share = len(matched) / len(headings) if headings else 0.0  # fraction of expected headings found
    chapter_lengths = [record.character_end - record.character_start for record in records]  # for the shortest and longest chapter
    report = {  # what the build report shows for this book
        "expected_headings": len(headings), "found_headings": len(matched), "aligned_share": round(aligned_share, 4),  # how many headings were expected and found
        "unmatched_headings": [{"chapter_number": number, "heading": headings[number]} for number, match in enumerate(matches) if match is None],  # which headings were not found
        "shortest_chapter_characters": min(chapter_lengths, default=0), "longest_chapter_characters": max(chapter_lengths, default=0),  # chapter length range
        "appendix_characters_without_chapter": len(combined_text) - story_end,  # text after the last chapter, cited by book only
        "gate_passed": aligned_share >= minimum_aligned_share,  # DESIGN.md 8.3: below the gate a book ships without chapter fields
    }
    return (records if report["gate_passed"] else []), report  # a failed book keeps no chapter records


def align_chapters(part_texts: Mapping[int, str], headings_by_book: Mapping[int, list[str]], minimum_aligned_share: float) -> ChapterAlignment:  # aligns every book
    all_records, report = [], {}  # kept chapters and per-book reports
    for book_index, headings in sorted(headings_by_book.items()):  # every book
        records, book_report = align_book(book_index, part_texts, headings, minimum_aligned_share)  # one book
        all_records.extend(records)  # keep its chapters
        report["book_%d" % book_index] = book_report  # report keyed by the book number of the pair locations
    all_records.sort(key=lambda record: (record.part_number, record.character_start))  # part by part, text order
    return ChapterAlignment(tuple(all_records), report)  # chapters and the report
