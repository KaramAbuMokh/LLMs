# Chapter alignment on invented books: forward matching, split headings, a lost heading, the appendix cut, a chapter crossing a part edge, the 95% gate
from retrieval_index_builder.source_alignment.chapter_alignment import MINIMUM_CHAPTER_CHARACTERS, align_chapters, find_chapter_at  # code under test

BODY_WORDS = "the grey wolf crossed the frozen river and the long night fell on the keep".split()  # filler words


def make_body(word_count: int, seed_word: str) -> str:  # chapter text long enough to pass the minimum chapter length
    return " ".join([seed_word] + [BODY_WORDS[index % len(BODY_WORDS)] for index in range(word_count)]) + "."  # a start word, then filler


def make_book(headings_and_words: list[tuple[str, int]]) -> str:  # "PROLOGUE We ..." blocks joined by spaces
    return " ".join("%s %s" % (heading, make_body(words, "We")) for heading, words in headings_and_words)  # heading, capital start word, body


def test_headings_are_found_in_order_with_split_letters_and_the_appendix_is_cut():  # book 1 (Storm) has one part
    book_text = make_book([("PROLOGUE", 700), ("J AIME", 700), ("CATELYN", 700)]) + " APPENDIX THE KINGS AND THEIR COURTS " + make_body(300, "House")  # three chapters, then an appendix
    alignment = align_chapters({1: book_text}, {1: ["PROLOGUE", "JAIME", "CATELYN"]}, 0.95)  # code under test
    assert [(chapter.chapter_number, chapter.chapter_title) for chapter in alignment.chapters] == [(0, "Prologue"), (1, "Jaime"), (2, "Catelyn")]  # all three
    assert alignment.chapters[0].character_start == len("PROLOGUE ")  # the chapter starts after its heading
    assert alignment.chapters[1].character_start == book_text.index("J AIME") + len("J AIME ")  # split heading handled
    assert alignment.chapters[0].character_end == book_text.index("J AIME")  # a chapter ends where the next heading begins
    assert alignment.chapters[2].character_end == book_text.index("APPENDIX")  # the last chapter stops at the appendix
    report = alignment.report["book_1"]  # per-book facts
    assert report["found_headings"] == 3 and report["gate_passed"] and report["appendix_characters_without_chapter"] > 0  # reported


def test_a_heading_missing_from_the_text_stays_unmatched_and_numbers_do_not_shift():  # TYRION is absent, JON must keep number 3
    book_text = make_book([("PROLOGUE", 700), ("CATELYN", 700), ("JON", 700)] * 1)  # no TYRION in the text
    headings = ["PROLOGUE", "CATELYN", "TYRION", "JON"]  # expected sequence
    alignment = align_chapters({1: book_text}, {1: headings}, 0.5)  # a low gate keeps the chapters
    assert [(chapter.chapter_number, chapter.chapter_title) for chapter in alignment.chapters] == [(0, "Prologue"), (1, "Catelyn"), (3, "Jon")]  # gap in numbers
    assert alignment.report["book_1"]["unmatched_headings"] == [{"chapter_number": 2, "heading": "TYRION"}]  # reported


def test_a_book_below_the_gate_keeps_no_chapters():  # 2 of 4 headings found with a 95% gate
    book_text = make_book([("PROLOGUE", 700), ("JAIME", 700)])  # only two headings exist
    alignment = align_chapters({1: book_text}, {1: ["PROLOGUE", "JAIME", "ARYA", "BRAN"]}, 0.95)  # code under test
    assert alignment.chapters == () and alignment.report["book_1"]["gate_passed"] is False  # book-only citations


def test_a_capitalised_name_right_after_a_heading_is_not_taken_for_the_next_heading():  # shouted names inside prose sit closer than the minimum chapter length
    false_hit = "JON We walked on. TYRION Lannister laughed at the joke. " + make_body(700, "Then")  # a stray "TYRION Lannister" 40 characters into Jon's chapter
    book_text = make_book([("PROLOGUE", 700)]) + " " + false_hit + " " + make_book([("TYRION", 700)])  # then the real Tyrion heading
    alignment = align_chapters({1: book_text}, {1: ["PROLOGUE", "JON", "TYRION"]}, 0.95)  # code under test
    tyrion = [chapter for chapter in alignment.chapters if chapter.chapter_title == "Tyrion"][0]  # the Tyrion chapter
    assert tyrion.character_start == book_text.rindex("TYRION") + len("TYRION ")  # the real heading, not the stray one
    assert MINIMUM_CHAPTER_CHARACTERS > book_text.index("TYRION Lannister") - book_text.index("JON We")  # the stray name was inside the minimum distance


def test_a_chapter_crossing_a_part_boundary_gets_one_record_per_part():  # Feast: parts 3, 6, 4 are one sequence
    first = make_book([("PROLOGUE", 700), ("CERSEI", 700)])  # part 3
    second_continuation = make_body(500, "Still")  # part 6 starts inside Cersei's chapter
    third = make_book([("JAIME", 700)])  # part 4 starts with a new heading
    part_texts = {3: first + " " + make_body(100, "Tail"), 6: second_continuation, 4: third}  # three parts of book 3
    alignment = align_chapters(part_texts, {3: ["PROLOGUE", "CERSEI", "JAIME"]}, 0.95)  # code under test
    cersei_pieces = [chapter for chapter in alignment.chapters if chapter.chapter_number == 1]  # Cersei's records
    assert [piece.part_number for piece in cersei_pieces] == [3, 6]  # one piece in part 3, one in part 6
    assert cersei_pieces[1].character_start == 0 and cersei_pieces[1].character_end == len(second_continuation)  # the whole of part 6
    assert [chapter.part_number for chapter in alignment.chapters if chapter.chapter_number == 2] == [4]  # Jaime only in part 4


def test_find_chapter_at_uses_half_open_spans_and_heading_gaps_belong_to_no_chapter():  # edges of chapter spans
    from retrieval_contracts.bundle_records import ChapterRecord  # record type for the hand-made chapters
    chapters = [ChapterRecord(1, 0, "Prologue", 10, 100), ChapterRecord(1, 1, "Jaime", 110, 200), ChapterRecord(2, 1, "Arya", 0, 50)]  # two parts
    assert find_chapter_at(chapters, 1, 10).chapter_title == "Prologue"  # first character
    assert find_chapter_at(chapters, 1, 99).chapter_title == "Prologue"  # last character
    assert find_chapter_at(chapters, 1, 100) is None  # end is exclusive and the next heading sits in the gap
    assert find_chapter_at(chapters, 1, 110).chapter_title == "Jaime"  # next chapter starts after its heading
    assert find_chapter_at(chapters, 2, 5).chapter_title == "Arya" and find_chapter_at(chapters, 3, 5) is None  # parts are separate
