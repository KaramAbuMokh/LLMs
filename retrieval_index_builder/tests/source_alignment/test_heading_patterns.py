# Heading patterns against the broken forms the cleaned books really have (invented sentences, no book text)
import pytest  # parametrised cases
from retrieval_index_builder.source_alignment.heading_patterns import build_heading_pattern, format_chapter_title  # code under test


@pytest.mark.parametrize("heading, text", [  # (expected heading, cleaned text that must match)
    ("CATELYN", "the door closed. C ATELYN Catelyn had never liked the place."),  # drop capital split off the first letter
    ("PROLOGUE", "PROLOGU E “We should start back,” the old man said."),  # split last letter and an opening quote after it
    ("BRIENNE", "the road was long.BRIENNE I t was a cold morning."),  # glued to the sentence before, split first letter after
    ("TYRION", "gone. TYRION He drank his way across the sea."),  # plain heading
    ("CHAPTER ONE ARYA", "CHAPTER ONE ARYA Arya was awake before the sun."),  # numbered chapter with its name
    ("CHAPTER ONE ARYA", "CHAPTER ONE 12ARYA Arya was awake before the sun."),  # page number between the number and the name
    ("THE KRAKEN’S DAUGHTER", "done. THE KRAKEN'S DAUGHTER The sea was black."),  # straight apostrophe for a curly one
])
def test_pattern_matches_broken_heading_forms(heading, text):  # each form is found
    assert build_heading_pattern(heading).pattern.search(text) is not None  # a match


@pytest.mark.parametrize("heading, text", [  # (expected heading, text that must NOT match)
    ("CATELYN", "she thought of Catelyn and wept."),  # ordinary mixed-case name
    ("JON", "JONATHAN Smith arrived at dusk."),  # inside a longer capitalised word
    ("REEK", "the CREEK Rose in spring."),  # tail of a longer capitalised word
    ("TYRION", "TYRION went home."),  # followed by a lowercase word, so not a chapter start
])
def test_pattern_refuses_look_alikes(heading, text):  # false hits are refused
    assert build_heading_pattern(heading).pattern.search(text) is None  # no match


def test_unnamed_numbered_chapter_reads_the_name_from_the_text():  # CHAPTER FOURTEEN has no name in the raw file
    heading_pattern = build_heading_pattern("CHAPTER FOURTEEN")  # pattern without a name
    match = heading_pattern.pattern.search("said Jon. CHAPTER FOURTEEN 112ARYA The river was blue.")  # cleaned text with a page number
    assert heading_pattern.title is None and match is not None and match.group("name") == "ARYA"  # the title comes from the match


def test_titles_are_title_cased_with_small_words_down():  # citation labels
    assert format_chapter_title("THE LOST LORD") == "The Lost Lord"  # plain
    assert format_chapter_title("A GHOST IN WINTERFELL") == "A Ghost in Winterfell"  # small word inside
    assert format_chapter_title("THE KRAKEN’S DAUGHTER") == "The Kraken’s Daughter"  # apostrophe
    assert format_chapter_title("PROLOGUE") == "Prologue"  # single word
    assert build_heading_pattern("CHAPTER ONE ARYA").title == "Arya"  # numbered chapter with a name
