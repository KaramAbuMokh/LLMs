# Reads the contents list at the top of a raw book (one chapter name per line, a few lines glued together) into separate headings
import re  # entry shapes
from typing import Sequence  # lines in

CONTENTS_ENTRY_PATTERN = re.compile(r"[A-Za-z][A-Za-z’'\-\., …:]*")  # a contents line: letters and light punctuation only
CONTENTS_CHECK_LINE_COUNT = 10  # lines after "Prologue" that must all look like entries for it to be a contents list
CONTENTS_SEARCH_LINE_COUNT = 400  # a contents list sits near the top of the file
MAX_ENTRY_CHARACTERS = 40  # longer lines are prose
CONTENTS_STOP_PREFIXES = ("APPENDIX", "THE KINGS AND THEIR COURTS", "MEANWHILE", "HOUSE ", "ACKNOWLEDGMENTS")  # what follows the last chapter
STANDALONE_HEADINGS = frozenset({"PROLOGUE", "EPILOGUE"})  # known names that are never glued to anything else
CAMEL_CASE_JOIN_PATTERN = re.compile(r"(?<=[a-z])(?=[A-Z])")  # "EddardTyrion": two names glued with a lower-to-upper step


def is_contents_entry(line: str) -> bool:  # True for a short line made only of letters and light punctuation
    return 0 < len(line) <= MAX_ENTRY_CHARACTERS and CONTENTS_ENTRY_PATTERN.fullmatch(line) is not None  # shape test


def find_contents_start(lines: Sequence[str]) -> int | None:  # index of the "Prologue" line that opens a contents list
    for line_index, line in enumerate(lines[:CONTENTS_SEARCH_LINE_COUNT]):  # only the top of the file
        if line.replace(" ", "").upper() != "PROLOGUE":  # not a lone prologue line
            continue  # keep looking
        following_lines = [following for following in lines[line_index + 1:] if following][:CONTENTS_CHECK_LINE_COUNT]  # next non-empty lines
        if len(following_lines) == CONTENTS_CHECK_LINE_COUNT and all(is_contents_entry(following) for following in following_lines):  # entries follow
            return line_index  # this is the list
    return None  # prose follows every prologue line: the file has no contents list


def split_camel_case_entries(entries: Sequence[str]) -> list[str]:  # "EddardTyrion" becomes "Eddard", "Tyrion"
    separated_entries = []  # entries after splitting
    for entry in entries:  # every entry in order
        if " " not in entry and CAMEL_CASE_JOIN_PATTERN.search(entry):  # one word with an inner lower-to-upper step
            separated_entries.extend(CAMEL_CASE_JOIN_PATTERN.split(entry))  # the glued names
        else:  # normal entry
            separated_entries.append(entry)  # keep it
    return separated_entries  # one name per entry


def split_upper_case_entries(entries: Sequence[str]) -> list[str]:  # "CATELYNJON" becomes "CATELYN", "JON" when both halves are names of the list
    known_names = {entry for entry in entries if " " not in entry} | STANDALONE_HEADINGS  # single-word names of this list
    separated_entries = []  # entries after splitting
    for entry in entries:  # every entry in order
        other_names = known_names - {entry}  # the glued entry itself does not count as a name
        glued_halves = next(  # first split point whose two halves are both known names
            ((entry[:cut], entry[cut:]) for cut in range(3, len(entry) - 2) if " " not in entry and entry[:cut] in other_names and entry[cut:] in other_names),  # first split point where both halves are known names
            None,  # default when no split point works
        )
        separated_entries.extend(glued_halves if glued_halves else [entry])  # two names, or the entry unchanged
    return separated_entries  # one name per entry


def read_contents_headings(lines: Sequence[str]) -> list[str]:  # upper-case chapter names of the contents list, or [] without one
    start_index = find_contents_start(lines)  # where the list opens
    if start_index is None:  # no contents list in this file
        return []  # the caller reads the body instead
    entries = []  # raw entries up to the appendix
    for line in lines[start_index:]:  # from "Prologue" onward
        if not line:  # blank lines inside the list
            continue  # skip
        if not is_contents_entry(line) or line.upper().startswith(CONTENTS_STOP_PREFIXES) or len(line) < 3:  # prose, appendix or a stray drop-cap letter
            break  # the chapters are over
        entries.append(line)  # a chapter entry
    return split_upper_case_entries([entry.upper() for entry in split_camel_case_entries(entries)])  # one upper-case name per chapter
