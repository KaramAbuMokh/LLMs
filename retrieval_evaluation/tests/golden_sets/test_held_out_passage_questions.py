# G1 on the shared fixture: question + gold answer through the locator, then the block split on synthetic centres (DESIGN.md §14.3)
from retrieval_contracts.rare_shingle_location import build_rare_shingle_index  # the shared locator
from retrieval_contracts.shared_fixture.fixture_files import load_fixture_golden_questions, load_fixture_part_texts  # fixture
from retrieval_evaluation.evaluation_settings import GoldenSetSettings  # block settings
from retrieval_evaluation.golden_sets.golden_question_records import DEVELOPMENT_SPLIT, TEST_SPLIT  # split names
from retrieval_evaluation.golden_sets.held_out_passage_questions import ValidationPair, assign_block_split, build_held_out_split  # under test

FIXTURE_SETTINGS = GoldenSetSettings(held_out_part_number=1, block_count=2, block_edge_margin_characters=10)  # part 1 is the "held-out" part of the fixture


def build_fixture_split():  # fixture questions through the G1 builder
    part_texts = load_fixture_part_texts()  # two invented parts
    pairs = [ValidationPair(golden_question.question_id, golden_question.question, golden_question.gold_answer) for golden_question in load_fixture_golden_questions() if golden_question.answerable]  # fg-1..fg-4
    return build_held_out_split(build_rare_shingle_index(part_texts), pairs, part_texts, FIXTURE_SETTINGS), part_texts  # split and texts


def test_fixture_questions_fg3_and_fg4_are_located_inside_their_gold_sentence():  # located with a hit in the gold span
    split, _ = build_fixture_split()  # built
    located = {question.question_id: question for question in split.development + split.test}  # kept questions
    gold_spans = {golden_question.question_id: golden_question.gold_spans[0] for golden_question in load_fixture_golden_questions() if golden_question.gold_spans}  # gold sentences
    assert set(located) <= {"fg-3", "fg-4"}  # fg-1 and fg-2 are dropped (not in the searched part, fewer than two close hits)
    assert split.answerable_count == 4 and split.located_count >= 1  # counts add up
    for question_id, question in located.items():  # every located question
        span = gold_spans[question_id]  # its gold sentence
        assert any(span.character_start <= hit.character_start and hit.character_end <= span.character_end for hit in question.gold_hits)  # a hit inside


def test_fg1_and_fg2_are_not_located_in_part_one():  # their answers live in part 0
    split, _ = build_fixture_split()  # built
    assert not {"fg-1", "fg-2"} & {golden_question.question_id for golden_question in split.development + split.test}  # dropped


def test_block_split_on_synthetic_centres():  # even blocks tune, odd blocks report, near-edge centres are dropped
    settings = GoldenSetSettings(block_count=10, block_edge_margin_characters=2000)  # as shipped
    length = 100000  # block width 10,000
    assert assign_block_split(5000, length, settings) == DEVELOPMENT_SPLIT  # block 0
    assert assign_block_split(15000, length, settings) == TEST_SPLIT  # block 1
    assert assign_block_split(25000, length, settings) == DEVELOPMENT_SPLIT  # block 2
    assert assign_block_split(11999, length, settings) is None  # 1,999 after the edge of block 1
    assert assign_block_split(12000, length, settings) == TEST_SPLIT  # exactly 2,000 away is kept
    assert assign_block_split(19000, length, settings) is None  # 1,000 before the next edge
    assert assign_block_split(1000, length, settings) is None  # near the start of the part
    assert assign_block_split(99999, length, settings) is None  # near the end of the part
