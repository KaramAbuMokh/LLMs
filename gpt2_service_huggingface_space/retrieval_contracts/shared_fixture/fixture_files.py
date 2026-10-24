# Paths and loaders for the shared test fixture: a tiny invented two-book corpus as a complete reference bundle, golden questions and expected numbers
import json  # fixture files
from dataclasses import dataclass  # frozen golden-question records
from pathlib import Path  # fixture paths
from typing import Any  # expectation values

FIXTURE_FOLDER = Path(__file__).resolve().parent  # this folder
FIXTURE_BUNDLE_FOLDER = FIXTURE_FOLDER / "fixture_bundle"  # a complete bundle in the real format (manifest + 7 data files)
FIXTURE_PART_TEXTS_PATH = FIXTURE_FOLDER / "fixture_part_texts.json"  # the invented part texts the passages were cut from
GOLDEN_QUESTIONS_PATH = FIXTURE_FOLDER / "fixture_golden_questions.json"  # questions with gold spans (two have no answer)
EXPECTATIONS_PATH = FIXTURE_FOLDER / "fixture_expectations.json"  # rankings, fusion, gate and metric values every package must reproduce


@dataclass(frozen=True)
class GoldSpan:  # where the answer sits: a character span of one part text (spans, not passage ids, survive re-chunking)
    part_number: int  # part the answer is in
    character_start: int  # first character of the answer sentence
    character_end: int  # one past its last character


@dataclass(frozen=True)
class FixtureGoldenQuestion:  # one test question of the fixture
    question_id: str  # "fg-<n>"
    question: str  # the question text
    answerable: bool  # False = the corpus does not answer it and the evidence gate must say so
    gold_spans: tuple[GoldSpan, ...]  # empty when not answerable
    gold_answer: str  # short reference answer ("" when not answerable)
    accepted_terms: tuple[str, ...]  # an answer is right when it contains any of these (lower case)


def load_fixture_part_texts() -> dict[int, str]:  # part number -> invented part text
    stored = json.loads(FIXTURE_PART_TEXTS_PATH.read_text(encoding="utf-8"))  # {"0": "...", "1": "..."}
    return {int(part_number): text for part_number, text in stored.items()}  # integer keys


def load_fixture_golden_questions() -> tuple[FixtureGoldenQuestion, ...]:  # every fixture question in file order
    stored = json.loads(GOLDEN_QUESTIONS_PATH.read_text(encoding="utf-8"))  # list of plain objects
    return tuple(  # frozen records
        FixtureGoldenQuestion(
            item["question_id"], item["question"], item["answerable"],  # identity and kind
            tuple(GoldSpan(**span) for span in item["gold_spans"]), item["gold_answer"], tuple(item["accepted_terms"]),  # gold
        )
        for item in stored  # every question
    )


def load_fixture_expectations() -> dict[str, Any]:  # the expected numbers (see the fixture AGENTS.md for each key)
    return json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))  # plain dictionary
