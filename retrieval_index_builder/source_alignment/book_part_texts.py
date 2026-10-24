# Loads the seven cleaned book part texts (six training parts and the held-out stretch) from the mixed run's dataset folder
import json  # book_train_parts.json
from pathlib import Path  # dataset folder
from retrieval_contracts.book_parts import BOOK_PARTS, BOOK_TRAIN_PARTS_FILE_NAME, HELD_OUT_PART_NUMBER, HELD_OUT_PASSAGE_FILE_NAME  # file names

TRAINING_PART_COUNT = 6  # parts 0-5 come from book_train_parts.json


def load_book_part_texts(source_data_folder: Path) -> dict[int, str]:  # part number -> cleaned part text, all seven parts
    stored_parts = json.loads((source_data_folder / BOOK_TRAIN_PARTS_FILE_NAME).read_text(encoding="utf-8"))  # [{"book", "part", "text"}]
    part_texts = {int(stored_part["part"]): stored_part["text"] for stored_part in stored_parts}  # training parts by number
    if sorted(part_texts) != list(range(TRAINING_PART_COUNT)):  # the file must hold exactly parts 0-5
        raise ValueError("book_train_parts.json holds parts %s, expected 0-%d" % (sorted(part_texts), TRAINING_PART_COUNT - 1))  # names only
    part_texts[HELD_OUT_PART_NUMBER] = (source_data_folder / HELD_OUT_PASSAGE_FILE_NAME).read_text(encoding="utf-8")  # the held-out stretch is part 6
    if sorted(part_texts) != sorted(book_part.part_number for book_part in BOOK_PARTS):  # the contract's table and the data must agree
        raise ValueError("loaded parts do not match the contract's part table")  # refuse a drifted dataset
    return part_texts  # seven texts
