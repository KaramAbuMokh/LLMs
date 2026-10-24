# Turns an expected heading ("TYRION", "CHAPTER ONE ARYA") into a pattern for the cleaned book text, where letters are sometimes split
import re  # pattern building
from dataclasses import dataclass  # frozen heading description

NUMBERED_HEADING_PATTERN = re.compile(r"CHAPTER ([A-Z]+(?:-[A-Z]+)?)(?: ([A-Z]+))?")  # "CHAPTER ELEVEN" and "CHAPTER ELEVEN THEON"
SMALL_TITLE_WORDS = frozenset({"a", "an", "and", "at", "for", "in", "of", "on", "the", "to"})  # kept lower case inside a title
APOSTROPHE_PATTERN = r"['’‘ʼ]"  # the books use curly apostrophes, the raw files straight ones
HEADING_START_GUARD = r"(?<![A-Z])"  # not the tail of a longer capitalised word
HEADING_END_GUARD = r"(?=\s+[“\"‘'(\[—]?[A-Z])"  # the chapter text follows: a space, then a capital (maybe split: "I t was") or an opening quote
PAGE_NUMBER_GAP = r"\s+(?:\d{1,4}\s?)?"  # a page number can sit between the number word and the name ("CHAPTER FOURTEEN 112ARYA")
UNKNOWN_NAME_GROUP = PAGE_NUMBER_GAP + r"(?P<name>[A-Z]{3,14})"  # the point-of-view name of a numbered chapter whose name the raw file lost


@dataclass(frozen=True)  # immutable record
class HeadingPattern:  # how to find one expected heading and what to call its chapter
    pattern: re.Pattern[str]  # matches the heading in the cleaned text
    title: str | None  # chapter title, None when it must be read from the match (group "name")


def format_chapter_title(heading_name: str) -> str:  # "THE LOST LORD" -> "The Lost Lord", "PROLOGUE" -> "Prologue"
    words = heading_name.lower().split(" ")  # lower-case words
    return " ".join(  # first letter up, small words down (never the first word)
        word if position > 0 and word in SMALL_TITLE_WORDS else word[:1].upper() + word[1:] for position, word in enumerate(words)  # capitalise each word, small words stay lower case
    )


def spell_out_letters(text: str) -> str:  # "TYRION" -> "T\s?Y\s?R\s?I\s?O\s?N": one optional space between any two letters
    pieces = []  # pattern pieces in order
    for character in text:  # every character of the heading
        if character == " ":  # word gap
            pieces.append(r"\s+")  # one or more spaces
        elif character in "'’":  # apostrophe in the heading
            pieces.append(APOSTROPHE_PATTERN)  # any apostrophe kind
        else:  # a letter or hyphen
            pieces.append(re.escape(character))  # itself
        pieces.append(r"\s?")  # a drop-cap split may follow any character
    return "".join(pieces[:-1])  # no optional space after the last character


def build_heading_pattern(heading: str) -> HeadingPattern:  # the pattern and title for one expected heading
    numbered_match = NUMBERED_HEADING_PATTERN.fullmatch(heading)  # Clash style?
    if numbered_match and not numbered_match.group(2):  # numbered chapter whose name the raw file lost
        pattern_text = spell_out_letters(heading) + UNKNOWN_NAME_GROUP  # name is captured from the cleaned text
        title = None  # read from the match
    elif numbered_match:  # numbered chapter with a known name
        pattern_text = spell_out_letters("CHAPTER " + numbered_match.group(1)) + PAGE_NUMBER_GAP + spell_out_letters(numbered_match.group(2))  # "CHAPTER ONE ARYA"
        title = format_chapter_title(numbered_match.group(2))  # the name is the title
    else:  # plain heading
        pattern_text = spell_out_letters(heading)  # the name itself
        title = format_chapter_title(heading)  # title-cased heading
    return HeadingPattern(re.compile(HEADING_START_GUARD + pattern_text + HEADING_END_GUARD), title)  # compiled, case-sensitive
