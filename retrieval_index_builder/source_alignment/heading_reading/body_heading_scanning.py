# Reads chapter headings from the BODY of a raw book that has no contents list: upper-case heading lines (Dance) or "CHAPTER <ORDINAL>" lines (Clash)
import re  # heading shapes
from typing import Sequence  # lines in

PROLOGUE_LINE_PATTERN = re.compile(r"(?:^|\s)PROLOGU ?E$")  # "PROLOGU E" and "April 2011 PROLOGUE": the first heading of the body
UPPER_CASE_HEADING_PATTERN = re.compile(r"[A-Z][A-Z’' \-\.]{1,43}[A-Z]")  # a whole line of capitals, 3 to 45 characters
GLUED_TAIL_PATTERN = re.compile(r".*[.!?”’\"…)] ([A-Z][A-Z’' \-]{1,40}[A-Z])")  # a heading glued to the end of the line before it ("... whispered. BRAN")
CHAPTER_HEADING_PATTERN = re.compile(r"CHAPTER ([A-Z]+(?:-[A-Z]+)?)")  # "CHAPTER TWENTY-ONE"
NAME_LINE_PATTERN = re.compile(r"[A-Z]{3,14}")  # the point-of-view name line under a "CHAPTER" line
LAST_HEADING = "EPILOGUE"  # the body headings of an upper-case-style book end here (an appendix follows)
ONES = ("", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN",  # number words 1-19
        "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN")  # number words 1-19
TENS = ("", "", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY")  # number words for multiples of ten


def spell_number(number: int) -> str:  # 21 -> "TWENTY-ONE", 30 -> "THIRTY", 12 -> "TWELVE" (numbers 1-99)
    if number < 20:  # one word
        return ONES[number]  # ONE ... NINETEEN
    return TENS[number // 10] + ("-" + ONES[number % 10] if number % 10 else "")  # tens word, then the ones word when there is one


def find_next_line(lines: Sequence[str], start_index: int) -> str:  # the next non-empty line at or after start_index ("" when there is none)
    return next((line for line in lines[start_index:] if line), "")  # first line with content


def scan_chapter_style_headings(lines: Sequence[str], first_index: int) -> list[str]:  # "CHAPTER ONE ARYA" style; missing numbers are filled in
    named_chapters: dict[int, str | None] = {}  # chapter number -> name line (None when the raw text lost it)
    for line_index in range(first_index, len(lines)):  # the body
        chapter_match = CHAPTER_HEADING_PATTERN.fullmatch(lines[line_index])  # a "CHAPTER <ORDINAL>" line
        if chapter_match and chapter_match.group(1) in SPELLED_NUMBERS:  # a real ordinal word
            following_line = find_next_line(lines, line_index + 1)  # the line under it
            named_chapters[SPELLED_NUMBERS[chapter_match.group(1)]] = following_line if NAME_LINE_PATTERN.fullmatch(following_line) else None  # its name
    headings = ["PROLOGUE"]  # the body opens with the prologue
    for number in range(1, max(named_chapters) + 1):  # every chapter number up to the last one found, so a lost line leaves no gap
        name = named_chapters.get(number)  # None when the raw text lost the number line or the name line
        headings.append("CHAPTER %s" % spell_number(number) + (" " + name if name else ""))  # name left out when unknown: read from the cleaned text later
    return headings  # prologue and numbered chapters


def scan_upper_case_headings(lines: Sequence[str], first_index: int) -> list[str]:  # one heading per whole-capital line, up to the epilogue
    headings = ["PROLOGUE"]  # the body opens with the prologue
    for line in lines[first_index + 1:]:  # lines after the prologue
        glued_tail = GLUED_TAIL_PATTERN.fullmatch(line)  # a heading at the end of a prose line
        heading = line if UPPER_CASE_HEADING_PATTERN.fullmatch(line) else glued_tail.group(1) if glued_tail else None  # the heading on this line, if any
        if heading:  # a heading line
            headings.append(heading)  # keep it
            if heading == LAST_HEADING:  # the appendix starts after this
                break  # stop
    return headings  # prologue, chapters and epilogue


def scan_body_headings(lines: Sequence[str]) -> list[str]:  # headings of a book without a contents list, in reading order
    first_index = next((index for index, line in enumerate(lines) if PROLOGUE_LINE_PATTERN.search(line)), None)  # the prologue line
    if first_index is None:  # nothing to read from
        return []  # the caller reports an empty list
    has_chapter_lines = any(CHAPTER_HEADING_PATTERN.fullmatch(line) for line in lines[first_index:])  # Clash style
    return scan_chapter_style_headings(lines, first_index) if has_chapter_lines else scan_upper_case_headings(lines, first_index)  # pick the style


SPELLED_NUMBERS = {spell_number(number): number for number in range(1, 100)}  # "TWENTY-ONE" -> 21
