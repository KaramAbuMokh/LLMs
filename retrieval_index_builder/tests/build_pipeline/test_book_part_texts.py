# Loading the seven part texts: the held-out stretch is part 6, and a dataset with a missing training part is refused
import json  # invented dataset files
import pytest  # expected errors
from retrieval_contracts.book_parts import BOOK_TRAIN_PARTS_FILE_NAME, HELD_OUT_PASSAGE_FILE_NAME  # file names
from retrieval_index_builder.source_alignment.book_part_texts import load_book_part_texts  # code under test


def write_dataset(folder, part_numbers) -> None:  # an invented dataset with the given training parts
    parts = [{"book": "Invented", "part": number, "text": "text of part %d" % number} for number in part_numbers]  # parts
    (folder / BOOK_TRAIN_PARTS_FILE_NAME).write_text(json.dumps(parts), encoding="utf-8")  # training parts file
    (folder / HELD_OUT_PASSAGE_FILE_NAME).write_text("held out text", encoding="utf-8")  # part 6


def test_seven_parts_are_loaded_with_the_held_out_stretch_as_part_six(tmp_path):  # the normal case
    write_dataset(tmp_path, range(6))  # parts 0-5
    part_texts = load_book_part_texts(tmp_path)  # code under test
    assert sorted(part_texts) == list(range(7)) and part_texts[3] == "text of part 3" and part_texts[6] == "held out text"  # all parts, right texts


def test_a_missing_training_part_is_refused(tmp_path):  # a drifted dataset must fail loudly
    write_dataset(tmp_path, [0, 1, 2, 3, 5])  # part 4 missing
    with pytest.raises(ValueError):  # must refuse
        load_book_part_texts(tmp_path)  # code under test
