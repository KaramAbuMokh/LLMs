# Turns the 34,001 cleaned training pairs into stored QuestionAnswerRecords with book and chapter
import json  # question_answer_train.json
from collections import Counter  # placement counts
from pathlib import Path  # pairs file
from typing import Mapping, Sequence  # inputs
from retrieval_contracts.book_parts import find_book_part  # book title and series number of a part
from retrieval_contracts.bundle_records import ChapterRecord, QuestionAnswerRecord  # stored record types
from retrieval_contracts.rare_shingle_location import build_rare_shingle_index  # shingle places of all parts
from retrieval_index_builder.index_build_settings import IndexBuildSettings  # window width
from retrieval_index_builder.source_alignment.pair_placement import place_pair  # book and chapter of one pair


def build_pair_records(pairs_path: Path, part_texts: Mapping[int, str], chapters: Sequence[ChapterRecord],  # stored pair records plus the counts of how they were placed
                       settings: IndexBuildSettings) -> tuple[tuple[QuestionAnswerRecord, ...], Mapping[str, int]]:  # records in file order and counts by method
    stored_pairs = json.loads(pairs_path.read_text(encoding="utf-8"))  # [{source_index, q, a, training_text, book_location}]
    part_lengths = {part_number: len(text) for part_number, text in part_texts.items()}  # characters per part
    shingle_index = build_rare_shingle_index(part_texts)  # about 5 seconds and 1 GB for the real books
    records, placement_counts = [], Counter()  # output and tally
    for stored_pair in stored_pairs:  # file order is the row order of the question vectors
        placement = place_pair(  # book and chapter by the shared rule
            stored_pair["q"], stored_pair["a"], stored_pair["book_location"], part_lengths, chapters, shingle_index, settings.pair_window_characters,  # question, answer, coarse location, part sizes, chapters, shingle places, window width
        )
        placement_counts[placement.method] += 1  # tally the rule that placed it
        book_part = find_book_part(placement.part_number) if placement.part_number is not None else None  # book of the placed part
        records.append(QuestionAnswerRecord(  # the stored pair
            pair_id="qa-%d" % stored_pair["source_index"], source_index=stored_pair["source_index"], question=stored_pair["q"], answer=stored_pair["a"],  # identity and text of the pair
            book_title=book_part.book_title if book_part else None, series_number=book_part.series_number if book_part else None,  # book of the placed part
            chapter_number=placement.chapter.chapter_number if placement.chapter else None, chapter_title=placement.chapter.chapter_title if placement.chapter else None,  # chapter of the placed span
        ))
    return tuple(records), dict(placement_counts)  # frozen records and a plain dictionary of counts
