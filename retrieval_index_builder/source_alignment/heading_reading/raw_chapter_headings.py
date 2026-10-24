# Reads the chapter headings of one raw book file in reading order: from its contents list when it has one, else from the body
import re  # line shapes
from pathlib import Path  # raw book file
from typing import Mapping  # file name table
from retrieval_index_builder.source_alignment.heading_reading.body_heading_scanning import scan_body_headings  # Dance and Clash style books
from retrieval_index_builder.source_alignment.heading_reading.contents_list_reading import read_contents_headings  # Storm, Game and Feast style books

RAW_BOOK_FILE_NAMES: Mapping[int, str] = {  # location_book_index -> raw file name inside the raw books folder
    0: "91c2bca52e010b6723465056b05512e4.txt",  # A Dance with Dragons
    1: "A Storm of Swords - George R. R. Martin_404.txt",  # A Storm of Swords
    2: "A-Game-Of-Thrones-by-George-R.R.-Martin.txt",  # A Game of Thrones
    3: "a_feast_for_crows_-_george_rr_martin.txt",  # A Feast for Crows
    4: "clash of kings.txt",  # A Clash of Kings
}
WHITESPACE_RUN_PATTERN = re.compile(r"\s+")  # tabs and repeated spaces in the raw files


def clean_raw_line(raw_line: str) -> str:  # one raw line with tabs and repeated spaces folded away
    return WHITESPACE_RUN_PATTERN.sub(" ", raw_line).strip()  # single spaces, no edge spaces


def read_raw_chapter_headings(raw_book_path: Path) -> list[str]:  # upper-case heading texts in reading order, contents list skipped
    lines = [clean_raw_line(raw_line) for raw_line in raw_book_path.read_text(encoding="utf-8", errors="replace").split("\n")]  # cleaned lines
    contents_headings = read_contents_headings(lines)  # the clean list of names, when the file has one
    return contents_headings if contents_headings else scan_body_headings(lines)  # otherwise read the headings where they stand in the body
