# G1: the 816 answerable held-out questions, located in the held-out stretch (part 6) by the shared rare-shingle locator, then cut into
# development and test questions by 10 contiguous blocks, dropping those near a block edge (DESIGN.md §11.1)
import json  # validation pairs and book parts
from dataclasses import dataclass  # frozen records
from pathlib import Path  # source files
from typing import Mapping, Sequence  # inputs
from retrieval_contracts.book_parts import BOOK_TRAIN_PARTS_FILE_NAME, HELD_OUT_PART_NUMBER, HELD_OUT_PASSAGE_FILE_NAME  # source file names
from retrieval_contracts.rare_shingle_location import RareShingleIndex, SearchWindow, locate_text  # the shared answer locator
from retrieval_evaluation.evaluation_settings import GoldenSetSettings  # block settings
from retrieval_evaluation.golden_sets.golden_question_records import DEVELOPMENT_SPLIT, TEST_SPLIT, HeldOutQuestion  # records

VALIDATION_PAIRS_FILE_NAME = "question_answer_validation.json"  # all 1,233 held-out pairs, 816 flagged answerable


@dataclass(frozen=True)
class ValidationPair:  # one answerable held-out pair
    question_id: str  # "g1-<source index>"
    question: str  # the question
    answer: str  # the gold answer written with it


@dataclass(frozen=True)
class HeldOutSplit:  # the outcome of locating and cutting the held-out questions
    development: tuple[HeldOutQuestion, ...]  # even blocks
    test: tuple[HeldOutQuestion, ...]  # odd blocks
    answerable_count: int  # questions offered
    located_count: int  # questions the locator placed in the window
    near_edge_count: int  # located questions dropped for sitting near a block edge


def load_book_part_texts(source_data_folder: Path) -> dict[int, str]:  # the seven part texts: 0-5 from the training parts, 6 the held-out stretch
    stored_parts = json.loads((source_data_folder / BOOK_TRAIN_PARTS_FILE_NAME).read_text(encoding="utf-8"))  # [{"book", "part", "text"}]
    part_texts = {part_number: stored["text"] for part_number, stored in enumerate(stored_parts)}  # part number = position in the file
    part_texts[HELD_OUT_PART_NUMBER] = (source_data_folder / HELD_OUT_PASSAGE_FILE_NAME).read_text(encoding="utf-8")  # the held-out stretch
    return part_texts  # keys 0..6


def load_answerable_validation_pairs(source_data_folder: Path) -> tuple[ValidationPair, ...]:  # the pairs flagged answerable (816)
    stored_pairs = json.loads((source_data_folder / VALIDATION_PAIRS_FILE_NAME).read_text(encoding="utf-8"))  # all held-out pairs
    return tuple(ValidationPair("g1-%s" % stored["source_index"], stored["q"], stored["a"]) for stored in stored_pairs if stored["answerable"])  # kept


def assign_block_split(center: int, part_length: int, settings: GoldenSetSettings) -> str | None:
    """DEVELOPMENT_SPLIT for an even block, TEST_SPLIT for an odd one, None when the centre is within the edge margin of its block."""
    block_width = part_length / settings.block_count  # characters per block
    block_number = min(int(center // block_width), settings.block_count - 1)  # block holding the centre
    distance_from_start = center - block_number * block_width  # characters since the block began
    if min(distance_from_start, block_width - distance_from_start) < settings.block_edge_margin_characters:  # too close to either edge
        return None  # dropped: near-twin questions could sit on both sides
    return DEVELOPMENT_SPLIT if block_number % 2 == 0 else TEST_SPLIT  # even blocks tune, odd blocks report


def build_held_out_split(shingle_index: RareShingleIndex, pairs: Sequence[ValidationPair], part_texts: Mapping[int, str],
                         settings: GoldenSetSettings) -> HeldOutSplit:  # locate every pair, then cut by block
    part_number = settings.held_out_part_number  # the one part searched
    window = SearchWindow(part_number, 0, len(part_texts[part_number]))  # the whole held-out stretch
    development, test, located_count, near_edge_count = [], [], 0, 0  # outcome so far
    for pair in pairs:  # every answerable pair
        located = locate_text(shingle_index, pair.question + " " + pair.answer, window)  # question + answer, as the gold rule says
        if located is None:  # fewer than two close rare hits
            continue  # unlocated: dropped
        located_count += 1  # counted
        center = (located.character_start + located.character_end) // 2  # middle of the gold span
        split = assign_block_split(center, len(part_texts[part_number]), settings)  # development, test or None
        if split is None:  # near a block edge
            near_edge_count += 1  # counted, dropped
            continue  # not used
        question = HeldOutQuestion(pair.question_id, pair.question, pair.answer, located.hits, center, split)  # the G1 record
        (development if split == DEVELOPMENT_SPLIT else test).append(question)  # file it
    return HeldOutSplit(tuple(development), tuple(test), len(pairs), located_count, near_edge_count)  # the cut
