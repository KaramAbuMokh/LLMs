# Frozen record types stored in the retrieval bundle (one JSON object per line), shared by the index builder, the service and the evaluation
import dataclasses  # field lists and dictionary conversion
import json  # one record per line
from dataclasses import dataclass  # frozen record classes
from enum import StrEnum  # evidence kinds as readable strings
from pathlib import Path  # bundle file paths
from typing import Any, Iterable, TypeVar  # generic record loading


class SourceType(StrEnum):  # the two kinds of evidence the bundle holds
    BOOK_PASSAGE = "book_passage"  # a chunk of one chapter of one book part
    QUESTION_ANSWER_PAIR = "question_answer_pair"  # one of the 34,001 cleaned training pairs


@dataclass(frozen=True)
class PassageRecord:  # one chunk of book text; line i of book_passages.jsonl is row i of the passage vectors
    passage_id: str  # stable id "bp-<part>-<chapter>-<chunk>", unique inside one bundle
    part_number: int  # key into BOOK_PARTS (6 = the held-out stretch of A Feast for Crows)
    book_title: str  # e.g. "A Storm of Swords"
    series_number: int  # publication order: 1 Game, 2 Clash, 3 Storm, 4 Feast, 5 Dance
    chapter_number: int | None  # 0 = prologue, 1.. = chapters in reading order, None when the chapter is unknown
    chapter_title: str | None  # point-of-view name, "Prologue" or "Epilogue"; None when unknown
    character_start: int  # first character of the passage inside its part text
    character_end: int  # one past the last character inside its part text
    token_count: int  # GPT-2 tokens of `text` (the prompt budget unit)
    text: str  # raw passage text (private: no public field ever carries more than 25 words of it)


@dataclass(frozen=True)
class QuestionAnswerRecord:  # one training pair; line i of question_answer_pairs.jsonl is row i of the question vectors
    pair_id: str  # "qa-<source_index>"
    source_index: int  # position of the pair in the original 61,650-pair file
    question: str  # question text (the only text behind the pair's vector)
    answer: str  # stored answer, written by another model: evidence for the prompt, never returned as the completion
    book_title: str | None  # book the pair came from, None when the pair was never located
    series_number: int | None  # publication order of that book, None when unlocated
    chapter_number: int | None  # chapter of the pair's rare-shingle location (DESIGN.md §8.5), else of a location window inside one chapter
    chapter_title: str | None  # that chapter's title, same rule


@dataclass(frozen=True)
class ChapterRecord:  # one chapter (or the piece of it inside one part) as a span of a part text
    part_number: int  # key into BOOK_PARTS
    chapter_number: int  # 0 = prologue, 1.. in reading order (numbering runs on across the three Feast parts)
    chapter_title: str  # point-of-view name, "Prologue" or "Epilogue"
    character_start: int  # first character of the chapter piece in the part text
    character_end: int  # one past its last character


BundleRecord = TypeVar("BundleRecord", PassageRecord, QuestionAnswerRecord, ChapterRecord)  # any of the three record types
FrozenRecord = TypeVar("FrozenRecord")  # any frozen dataclass rebuilt from stored values


def build_strict_record(record_type: type[FrozenRecord], values: dict[str, Any]) -> FrozenRecord:  # rebuilds a dataclass, refusing missing or unknown keys
    expected_names = {field.name for field in dataclasses.fields(record_type)}  # fields the type declares
    if set(values) != expected_names:  # any difference means the stored format and the code disagree
        raise ValueError("%s has keys %s, expected %s" % (record_type.__name__, sorted(values), sorted(expected_names)))  # names only, never text
    return record_type(**values)  # the frozen record


def record_to_json_line(record: PassageRecord | QuestionAnswerRecord | ChapterRecord) -> str:  # one JSONL line with sorted keys
    return json.dumps(dataclasses.asdict(record), ensure_ascii=False, sort_keys=True)  # stable text, so rebuilt files hash the same


def record_from_json_line(record_type: type[BundleRecord], line: str) -> BundleRecord:  # strict parse of one stored line
    return build_strict_record(record_type, json.loads(line))  # keys must match the dataclass exactly


def write_json_lines(path: Path, records: Iterable[PassageRecord | QuestionAnswerRecord | ChapterRecord]) -> int:  # writes a JSONL file, returns the line count
    line_count = 0  # lines written so far
    with path.open("w", encoding="utf-8", newline="\n") as output_file:  # Unix line ends on every machine
        for record in records:  # in the given order (the row order of the matching vectors)
            output_file.write(record_to_json_line(record) + "\n")  # one record per line
            line_count += 1  # count it
    return line_count  # how many records were written


def read_json_lines(path: Path, record_type: type[BundleRecord]) -> tuple[BundleRecord, ...]:  # reads a JSONL file back in row order
    with path.open("r", encoding="utf-8") as input_file:  # the stored records
        return tuple(record_from_json_line(record_type, line) for line in input_file if line.strip())  # skip only a trailing empty line
