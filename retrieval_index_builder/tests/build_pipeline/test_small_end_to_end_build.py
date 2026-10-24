# A whole small build on invented books with the hashing encoder: part texts -> chapters -> pairs -> passages -> bundle -> own manifest check
import dataclasses  # settings variants
import json  # invented dataset files
import numpy  # vector shape checks
from retrieval_contracts.book_parts import BOOK_TRAIN_PARTS_FILE_NAME, HELD_OUT_PASSAGE_FILE_NAME, find_book_part  # file names and part table
from retrieval_contracts.bundle_manifest import MANIFEST_FILE_NAME, PAIR_VECTORS_FILE_NAME, PASSAGE_VECTORS_FILE_NAME, find_bundle_problems, manifest_from_json  # checks
from retrieval_contracts.bundle_records import PassageRecord, read_json_lines  # read back the passages
from retrieval_contracts.shared_fixture.fixture_encoders import HashingTextEncoder  # fake encoder
from retrieval_index_builder.bundle_writing.bundle_assembly import write_index_bundle  # code under test
from retrieval_index_builder.index_build_settings import build_default_settings  # base settings
from retrieval_index_builder.passage_chunking.chapter_chunking import chunk_all_parts  # code under test
from retrieval_index_builder.source_alignment.book_part_texts import load_book_part_texts  # code under test
from retrieval_index_builder.source_alignment.chapter_alignment import align_chapters  # code under test
from retrieval_index_builder.source_alignment.pair_records import build_pair_records  # code under test

SENTENCE = "The grey wolf number %d crossed the frozen river near keep %d."  # invented sentence


def make_chapter(heading: str, first_number: int) -> str:  # a heading and 300 invented sentences, long enough to pass the minimum chapter length
    return heading + " We began. " + " ".join(SENTENCE % (number, number * 3) for number in range(first_number, first_number + 300))


def write_invented_dataset(folder) -> None:  # the files load_book_part_texts and build_pair_records read, with invented books in the 7 part slots
    part_texts = {number: "" for number in range(7)}  # every slot must exist
    part_texts[1] = make_chapter("PROLOGUE", 0) + " " + make_chapter("JAIME", 1000)  # Storm slot: two chapters
    part_texts[2] = make_chapter("PROLOGUE", 5000)  # Game slot: one chapter
    parts = [{"book": find_book_part(number).book_title, "part": number, "text": part_texts[number]} for number in range(6)]  # parts 0-5
    (folder / BOOK_TRAIN_PARTS_FILE_NAME).write_text(json.dumps(parts), encoding="utf-8")  # training parts
    (folder / HELD_OUT_PASSAGE_FILE_NAME).write_text(part_texts[6], encoding="utf-8")  # the held-out slot
    storm_offset = find_book_part(1).cleaned_start_offset + part_texts[1].index("number 1100 ")  # a coarse location inside Jaime's chapter
    pairs = [  # one located pair, one unlocated pair that the text answers, one that nothing answers
        {"source_index": 7, "q": "Where did the wolf cross?", "a": SENTENCE % (1100, 3300) + " " + SENTENCE % (1101, 3303), "training_text": "", "book_location": {"book": 1, "offset": storm_offset}},
        {"source_index": 8, "q": "Which keep?", "a": SENTENCE % (5100, 15300) + " " + SENTENCE % (5101, 15303), "training_text": "", "book_location": None},
        {"source_index": 9, "q": "Who sings?", "a": "Nobody sings in these invented books at all.", "training_text": "", "book_location": None},
    ]
    (folder / "question_answer_train.json").write_text(json.dumps(pairs), encoding="utf-8")  # pairs file


def test_small_build_produces_a_consistent_bundle(tmp_path):  # every stage feeds the next and the bundle checks out
    data_folder = tmp_path / "data"  # invented dataset
    data_folder.mkdir()  # create it
    write_invented_dataset(data_folder)  # fill it
    settings = dataclasses.replace(build_default_settings(), source_data_folder=data_folder, output_folder=tmp_path / "bundle", embedding_batch_size=16)  # test settings
    part_texts = load_book_part_texts(data_folder)  # seven parts
    alignment = align_chapters(part_texts, {1: ["PROLOGUE", "JAIME"], 2: ["PROLOGUE"]}, settings.minimum_aligned_share)  # chapters of two books
    pairs, placement_counts = build_pair_records(data_folder / "question_answer_train.json", part_texts, alignment.chapters, settings)  # pairs with places
    passages = chunk_all_parts(part_texts, alignment.chapters, settings)  # passages
    manifest = write_index_bundle(settings.output_folder, passages, pairs, alignment.chapters, HashingTextEncoder(), settings, [data_folder / "question_answer_train.json"])  # bundle
    assert placement_counts == {"shingles_in_window": 1, "shingles_corpus_wide": 1, "unplaced": 1}  # each pair placed by its own rule
    assert [(pair.book_title, pair.chapter_title) for pair in pairs] == [("A Storm of Swords", "Jaime"), ("A Game of Thrones", "Prologue"), (None, None)]  # books and chapters
    assert find_bundle_problems(settings.output_folder, manifest_from_json((settings.output_folder / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))) == []  # consistent
    stored_passages = read_json_lines(settings.output_folder / "book_passages.jsonl", PassageRecord)  # read back
    assert numpy.load(settings.output_folder / PASSAGE_VECTORS_FILE_NAME).shape == (len(stored_passages), 256)  # one vector per passage
    assert numpy.load(settings.output_folder / PAIR_VECTORS_FILE_NAME).shape == (3, 256)  # one vector per pair
    assert manifest.source_data_sha256.keys() == {"question_answer_train.json"} and manifest.passage_count == len(passages)  # manifest facts
