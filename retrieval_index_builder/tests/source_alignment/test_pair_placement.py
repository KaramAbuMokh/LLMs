# Pair placement on invented parts: each of the five methods, a window clamped at a part edge, a span centre on a chapter edge, never placed outside the window
import pytest  # parametrised cases
from retrieval_contracts.book_parts import find_book_part  # cleaned start offsets of the parts
from retrieval_contracts.bundle_records import ChapterRecord  # hand-made chapters
from retrieval_contracts.rare_shingle_location import build_rare_shingle_index  # shingle places
from retrieval_index_builder.source_alignment.pair_placement import build_search_window, place_pair  # code under test

PART_ONE_WORDS = ["ka%d" % number for number in range(3000)]  # part 1 (book 1): every 4-word run is unique
PART_TWO_WORDS = ["kb%d" % number for number in range(3000)]  # part 2 (book 2): another unique vocabulary
PART_TEXTS = {number: "" for number in range(7)} | {1: " ".join(PART_ONE_WORDS), 2: " ".join(PART_TWO_WORDS)}  # seven parts, two with text
PART_LENGTHS = {number: len(text) for number, text in PART_TEXTS.items()}  # characters per part
SHINGLE_INDEX = build_rare_shingle_index({1: PART_TEXTS[1], 2: PART_TEXTS[2]})  # places of every rare shingle
CHAPTERS = [ChapterRecord(1, 1, "Alpha", 0, 3000), ChapterRecord(1, 2, "Beta", 3000, 9000), ChapterRecord(1, 3, "Gamma", 9000, len(PART_TEXTS[1])),  # chapters of part 1 and one chapter of part 2
            ChapterRecord(2, 1, "Delta", 0, len(PART_TEXTS[2]))]  # edges at 3000 and 9000 in part 1
WINDOW = 1500  # half-width used by the tests (smaller than the real 20,000 so the invented parts stay short)


def slice_words(words: list[str], first: int, count: int) -> str:  # a run of consecutive part words, used as an "answer"
    return " ".join(words[first:first + count])  # exactly as the part text spells it


def pair_location(part_number: int, character_offset: int) -> dict[str, int]:  # a coarse book_location that maps to the given place
    book_part = find_book_part(part_number)  # table row of the part
    return {"book": book_part.location_book_index, "offset": book_part.cleaned_start_offset + character_offset}  # offset in the cleaned full book


def test_shingles_in_window_gives_part_and_the_chapter_of_the_span_centre():  # method 1
    answer = slice_words(PART_ONE_WORDS, 700, 10)  # words 700-709, about 4,200 characters in: inside chapter 2
    centre = PART_TEXTS[1].index(" " + PART_ONE_WORDS[700] + " ")  # roughly where the span sits
    placement = place_pair("Who?", answer, pair_location(1, centre), PART_LENGTHS, CHAPTERS, SHINGLE_INDEX, WINDOW)  # code under test
    assert (placement.part_number, placement.chapter.chapter_title, placement.method) == (1, "Beta", "shingles_in_window")  # placed by the text


def test_shingles_corpus_wide_when_the_pair_has_no_usable_location():  # method 2 (no location, or a location in trimmed matter)
    answer = slice_words(PART_TWO_WORDS, 100, 10)  # part 2 text
    for book_location in (None, {"book": 0, "offset": 5}):  # never located, and pointing into Dance front matter that was cut
        placement = place_pair("Who?", answer, book_location, PART_LENGTHS, CHAPTERS, SHINGLE_INDEX, WINDOW)  # code under test
        assert (placement.part_number, placement.chapter.chapter_title, placement.method) == (2, "Delta", "shingles_corpus_wide")  # found anywhere


def test_window_inside_chapter_when_no_shingles_but_the_window_fits_one_chapter():  # method 3
    placement = place_pair("Nothing matches this question.", "No shared words at all here.", pair_location(1, 6000), PART_LENGTHS, CHAPTERS, SHINGLE_INDEX, 500)  # a pair whose words occur nowhere, with a small window
    assert (placement.part_number, placement.chapter.chapter_title, placement.method) == (1, "Beta", "window_inside_chapter")  # window 5,500-6,500 sits in Beta


def test_window_only_when_no_shingles_and_the_window_spans_two_chapters():  # method 4
    placement = place_pair("Nothing matches this question.", "No shared words at all here.", pair_location(1, 3000), PART_LENGTHS, CHAPTERS, SHINGLE_INDEX, 500)  # the window straddles the Alpha/Beta edge
    assert (placement.part_number, placement.chapter, placement.method) == (1, None, "window_only")  # book only


def test_unplaced_without_location_and_without_shingles():  # method 5
    placement = place_pair("Nothing matches this question.", "No shared words at all here.", None, PART_LENGTHS, CHAPTERS, SHINGLE_INDEX, WINDOW)  # no location and no matching words
    assert (placement.part_number, placement.chapter, placement.method) == (None, None, "unplaced")  # nothing known


def test_a_pair_with_a_window_is_never_placed_outside_it():  # the text lives in part 2, the coarse location says part 1
    answer = slice_words(PART_TWO_WORDS, 100, 10)  # part 2 text
    placement = place_pair("Who?", answer, pair_location(1, 6000), PART_LENGTHS, CHAPTERS, SHINGLE_INDEX, 500)  # location in part 1
    assert placement.part_number == 1 and placement.method == "window_inside_chapter"  # stays in its window's part, found nothing there
    far_answer = slice_words(PART_ONE_WORDS, 10, 10)  # words 10-19 of part 1, about 50 characters in
    far_placement = place_pair("Who?", far_answer, pair_location(1, 8000), PART_LENGTHS, CHAPTERS, SHINGLE_INDEX, 500)  # window far away from them
    assert far_placement.method != "shingles_in_window"  # the text is in the same part but outside the window


def test_window_is_clamped_at_the_part_edges():  # a location near the start or end of a part
    near_start = build_search_window(pair_location(1, 100), PART_LENGTHS, WINDOW)  # 100 characters in
    near_end = build_search_window(pair_location(1, PART_LENGTHS[1] - 50), PART_LENGTHS, WINDOW)  # 50 characters from the end
    assert (near_start.character_start, near_start.character_end) == (0, 100 + WINDOW)  # clamped at 0
    assert (near_end.character_start, near_end.character_end) == (PART_LENGTHS[1] - 50 - WINDOW, PART_LENGTHS[1])  # clamped at the part end


@pytest.mark.parametrize("first_word, expected_title", [(505, "Alpha"), (512, "Alpha"), (515, "Beta"), (525, "Beta")])  # spans before, across (two ways) and after the chapter edge at 3,000
def test_span_centre_decides_the_chapter_at_a_chapter_edge(first_word, expected_title):  # centre, not start or end, picks the chapter
    answer = slice_words(PART_ONE_WORDS, first_word, 10)  # a 10-word span
    placement = place_pair("Who?", answer, pair_location(1, 3000), PART_LENGTHS, CHAPTERS, SHINGLE_INDEX, WINDOW)  # code under test
    span_start = PART_TEXTS[1].index(" " + PART_ONE_WORDS[first_word] + " ") + 1  # where the answer begins in part 1
    centre = span_start + len(answer) // 2  # its middle
    assert placement.method == "shingles_in_window" and placement.chapter.chapter_title == ("Alpha" if centre < 3000 else "Beta")  # chapter holding the centre
    assert placement.chapter.chapter_title == expected_title  # and the case is on the intended side of the edge
