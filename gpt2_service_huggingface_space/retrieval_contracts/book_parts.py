# The seven book parts the index reads (six training parts plus the held-out stretch of A Feast for Crows) and how pair locations map into them
from dataclasses import dataclass  # frozen table rows
from typing import Mapping  # part lengths by part number

BOOK_TRAIN_PARTS_FILE_NAME = "book_train_parts.json"  # parts 0-5: [{"book", "part", "text"}] in the mixed run's dataset folder
HELD_OUT_PASSAGE_FILE_NAME = "books_validation_text.txt"  # part 6: the held-out Feast passage, same folder
HELD_OUT_PART_NUMBER = 6  # part number given to the held-out passage


@dataclass(frozen=True)
class BookPart:  # one part text of one book
    part_number: int  # position in book_train_parts.json (0-5), or 6 for the held-out passage file
    book_title: str  # title shown in citations
    series_number: int  # publication order: 1 Game, 2 Clash, 3 Storm, 4 Feast, 5 Dance
    location_book_index: int  # the book number every pair's book_location uses (sorted raw file names: 0 Dance ... 4 Clash)
    cleaned_start_offset: int  # where this part starts inside the cleaned full book that book_location offsets point into
    reading_order: int  # order of the part inside its book (only A Feast for Crows has three parts)


BOOK_PARTS: tuple[BookPart, ...] = (  # every part; offsets from build_report.json (front cuts) and the split report (held-out span)
    BookPart(0, "A Dance with Dragons", 5, 0, 2357, 0),  # front matter cut: 2,357 characters
    BookPart(1, "A Storm of Swords", 3, 1, 2104, 0),  # front matter cut: 2,104
    BookPart(2, "A Game of Thrones", 1, 2, 769, 0),  # front matter cut: 769
    BookPart(3, "A Feast for Crows", 4, 3, 4039, 0),  # Feast before the held-out passage (front cut 4,039)
    BookPart(4, "A Feast for Crows", 4, 3, 765969, 2),  # Feast after the held-out passage (starts at cleaned offset 765,969)
    BookPart(5, "A Clash of Kings", 2, 4, 37, 0),  # front matter cut: 37
    BookPart(6, "A Feast for Crows", 4, 3, 504717, 1),  # the held-out passage, cleaned offsets 504,717 to 765,969
)


def find_book_part(part_number: int) -> BookPart:  # the table row for one part number
    for book_part in BOOK_PARTS:  # seven rows, a scan is enough
        if book_part.part_number == part_number:  # found it
            return book_part  # the row
    raise KeyError("unknown book part %d" % part_number)  # a record points at a part that does not exist


def locate_cleaned_offset(location_book_index: int, cleaned_offset: int, part_lengths: Mapping[int, int]) -> tuple[int, int] | None:
    """Maps a pair's book_location (book index, offset in the cleaned full book) to (part_number, offset inside that part).

    part_lengths holds the character length of every part text. Returns None when the offset falls in trimmed front or back matter.
    """
    for book_part in BOOK_PARTS:  # the parts of that book are the only candidates
        if book_part.location_book_index != location_book_index:  # another book
            continue  # skip it
        offset_in_part = cleaned_offset - book_part.cleaned_start_offset  # distance from the part's first character
        if 0 <= offset_in_part < part_lengths[book_part.part_number]:  # inside this part's text
            return book_part.part_number, offset_in_part  # the part and the local offset
    return None  # the offset points at text the cleaning removed
