# Self-test of the frozen contract against its own fixture (run: python -m pytest retrieval_contracts from the service folder)
import json  # strictness check input
import numpy  # vector comparison
import pytest  # expected errors
from retrieval_contracts.book_parts import BOOK_PARTS, locate_cleaned_offset  # pair location mapping
from retrieval_contracts.bundle_manifest import (  # manifest helpers and file names
    MANIFEST_FILE_NAME, PAIR_RECORDS_FILE_NAME, PAIR_VECTORS_FILE_NAME, PASSAGE_RECORDS_FILE_NAME, PASSAGE_VECTORS_FILE_NAME,
    find_bundle_problems, manifest_from_json, manifest_to_json,
)
from retrieval_contracts.bundle_records import PassageRecord, QuestionAnswerRecord, read_json_lines, record_from_json_line  # records
from retrieval_contracts.indexed_text import compose_indexed_passage_text, format_chapter_label  # indexed text rules
from retrieval_contracts.lexical_tokens import split_lexical_tokens  # word splitting
from retrieval_contracts.rare_shingle_location import SearchWindow, build_rare_shingle_index, locate_text  # shared answer locator
from retrieval_contracts.shared_fixture.fixture_encoders import HashingTextEncoder  # fake encoder
from retrieval_contracts.shared_fixture.fixture_files import (  # fixture loaders
    FIXTURE_BUNDLE_FOLDER, load_fixture_expectations, load_fixture_golden_questions, load_fixture_part_texts,
)

REAL_PART_LENGTHS = {0: 2305781, 1: 2287685, 2: 1601052, 3: 500678, 4: 897091, 5: 1764653, 6: 261252}  # characters per real part


def load_fixture_manifest():  # the fixture bundle's manifest
    return manifest_from_json((FIXTURE_BUNDLE_FOLDER / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))  # parsed strictly


def test_fixture_bundle_matches_its_manifest():  # every checksum, size and the format version agree
    assert find_bundle_problems(FIXTURE_BUNDLE_FOLDER, load_fixture_manifest()) == []  # no problems


def test_manifest_round_trip_is_stable():  # writing a parsed manifest gives the same text back
    manifest_text = (FIXTURE_BUNDLE_FOLDER / MANIFEST_FILE_NAME).read_text(encoding="utf-8")  # stored text
    assert manifest_to_json(manifest_from_json(manifest_text)) == manifest_text  # byte-identical


def test_passage_spans_point_at_their_text():  # record.text is exactly part_text[start:end]
    part_texts = load_fixture_part_texts()  # invented part texts
    for record in read_json_lines(FIXTURE_BUNDLE_FOLDER / PASSAGE_RECORDS_FILE_NAME, PassageRecord):  # every passage
        assert part_texts[record.part_number][record.character_start:record.character_end] == record.text  # exact span


def test_gold_spans_fall_inside_one_passage():  # each golden answer sentence sits inside a fixture passage
    passages = read_json_lines(FIXTURE_BUNDLE_FOLDER / PASSAGE_RECORDS_FILE_NAME, PassageRecord)  # all passages
    for golden_question in load_fixture_golden_questions():  # every question
        for span in golden_question.gold_spans:  # every gold span
            assert any(record.part_number == span.part_number and record.character_start <= span.character_start  # same part, starts inside
                       and span.character_end <= record.character_end for record in passages)  # and ends inside


def test_lexical_tokens_match_expectations():  # the tokenizer reproduces the recorded examples
    for text, expected_tokens in load_fixture_expectations()["tokens"].items():  # every example
        assert split_lexical_tokens(text) == expected_tokens  # same tokens in the same order


def test_fixture_encoder_reproduces_stored_vectors():  # the stored vectors came from the fake encoder and the contract's indexed text
    encoder = HashingTextEncoder()  # fake encoder
    passages = read_json_lines(FIXTURE_BUNDLE_FOLDER / PASSAGE_RECORDS_FILE_NAME, PassageRecord)  # passages
    pairs = read_json_lines(FIXTURE_BUNDLE_FOLDER / PAIR_RECORDS_FILE_NAME, QuestionAnswerRecord)  # pairs
    passage_vectors = numpy.load(FIXTURE_BUNDLE_FOLDER / PASSAGE_VECTORS_FILE_NAME, allow_pickle=False)  # stored passage vectors
    question_vectors = numpy.load(FIXTURE_BUNDLE_FOLDER / PAIR_VECTORS_FILE_NAME, allow_pickle=False)  # stored question vectors
    assert numpy.allclose(passage_vectors, encoder.encode_texts([compose_indexed_passage_text(r) for r in passages]))  # passages
    assert numpy.allclose(question_vectors, encoder.encode_texts([r.question for r in pairs]))  # questions alone


def test_pair_offsets_map_into_the_right_part():  # book_location offsets land in the right part, trimmed matter gives None
    assert locate_cleaned_offset(3, 504716, REAL_PART_LENGTHS) == (3, 500677)  # last character before the held-out passage
    assert locate_cleaned_offset(3, 504717, REAL_PART_LENGTHS) == (6, 0)  # first held-out character
    assert locate_cleaned_offset(3, 765969, REAL_PART_LENGTHS) == (4, 0)  # first character after the held-out passage
    assert locate_cleaned_offset(0, 100, REAL_PART_LENGTHS) is None  # Dance front matter was cut
    assert len({part.part_number for part in BOOK_PARTS}) == 7  # seven distinct parts


def test_records_refuse_unknown_keys():  # a format drift fails loudly instead of loading half a record
    stored = json.loads((FIXTURE_BUNDLE_FOLDER / PASSAGE_RECORDS_FILE_NAME).read_text(encoding="utf-8").splitlines()[0])  # one record
    stored["unexpected"] = 1  # add a key the contract does not know
    with pytest.raises(ValueError):  # strict parse
        record_from_json_line(PassageRecord, json.dumps(stored))  # must refuse


def test_chapter_labels():  # citation labels for numbered, unnumbered and unknown chapters
    assert format_chapter_label(12, "Tyrion") == "Tyrion (chapter 12)"  # numbered point-of-view chapter
    assert format_chapter_label(0, "Prologue") == "Prologue"  # prologue has no number
    assert format_chapter_label(None, None) is None  # unknown chapter


def test_rare_shingles_locate_answers_inside_their_gold_sentences():  # the shared locator finds two pair answers where the fixture says
    index = build_rare_shingle_index(load_fixture_part_texts())  # rare shingles of the invented parts
    pairs = {record.pair_id: record for record in read_json_lines(FIXTURE_BUNDLE_FOLDER / PAIR_RECORDS_FILE_NAME, QuestionAnswerRecord)}  # by id
    gold_spans = {question.question_id: question.gold_spans[0] for question in load_fixture_golden_questions() if question.gold_spans}  # by id
    for pair_id, question_id in (("qa-101", "fg-1"), ("qa-103", "fg-3")):  # pairs whose answer reuses the book's words
        located = locate_text(index, pairs[pair_id].answer)  # corpus-wide search
        gold_span = gold_spans[question_id]  # the answer sentence
        assert located is not None and located.part_number == gold_span.part_number  # found, in the right part
        assert gold_span.character_start <= located.character_start and located.character_end <= gold_span.character_end  # inside it


def test_rare_shingles_need_two_hits_inside_the_window():  # a search window outside the answer, or a single rare hit, gives no location
    index = build_rare_shingle_index(load_fixture_part_texts())  # rare shingles of the invented parts
    corin_answer = "Corin keeps the tower keys in a cedar box under the salt stairs."  # the answer sits in part 1
    assert locate_text(index, corin_answer, SearchWindow(1, 0, 400)) is not None  # found inside a window around it
    assert locate_text(index, corin_answer, SearchWindow(0, 0, 400)) is None  # not found inside another part's window
    assert locate_text(index, "the silver blade Dawnlight") is None  # exactly one rare shingle is not a location
