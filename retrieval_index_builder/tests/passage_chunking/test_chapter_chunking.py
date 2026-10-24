# Passage chunking on invented chapters: exact substrings, ceilings for the three chunk sizes, overlap rule, tail merge, chapter edges, unique ids
import dataclasses  # settings variants
import pytest  # parametrised sizes
from retrieval_contracts.bundle_records import ChapterRecord  # chapter spans
from retrieval_contracts.indexed_text import format_chapter_label  # the chapter label of public citations
from retrieval_index_builder.index_build_settings import build_default_settings  # base settings
from retrieval_index_builder.passage_chunking.chapter_chunking import chunk_all_parts, chunk_chapter  # code under test
from retrieval_index_builder.passage_chunking.gpt2_token_counting import count_gpt2_tokens  # token unit
from retrieval_index_builder.passage_chunking.sentence_spans import split_sentence_spans  # sentence boundaries

SIZES = [(128, 150), (200, 230), (256, 290)]  # (target, ceiling) of the three bundles


def make_sentence(number: int) -> str:  # sentences of varying length, all invented
    return "The grey wolf number %d crossed the frozen river near keep %d%s." % (number, number * 7, " and then the long night fell on the sleeping hills" * (number % 4))  # a sentence whose length depends on its number


def make_chapter_text(sentence_count: int) -> str:  # sentences joined by single spaces, like the cleaned books
    return " ".join(make_sentence(number) for number in range(sentence_count))  # all sentences in one text


def settings_for(target_tokens: int, max_tokens: int, **overrides):  # build settings with a chunk size
    return dataclasses.replace(build_default_settings(), target_tokens=target_tokens, max_tokens=max_tokens, **overrides)  # the default settings with the chunk size under test


@pytest.mark.parametrize("target_tokens, max_tokens", SIZES)  # the three bundles
def test_every_passage_is_an_exact_substring_inside_its_chapter_under_the_ceiling(target_tokens, max_tokens):  # core invariants
    chapter_text = "HEAD " + make_chapter_text(400)  # a chapter that starts after a heading
    chapter = ChapterRecord(1, 3, "Arya", 5, len(chapter_text))  # span after the heading
    passages = chunk_chapter(chapter_text, chapter, settings_for(target_tokens, max_tokens))  # code under test
    assert len(passages) > 3  # the chapter is long
    for passage in passages:  # every passage
        assert chapter_text[passage.character_start:passage.character_end] == passage.text  # exact substring
        assert chapter.character_start <= passage.character_start and passage.character_end <= chapter.character_end  # never leaves the chapter
        assert passage.token_count == count_gpt2_tokens(passage.text) <= max_tokens  # honest count under the ceiling
        assert (passage.chapter_number, passage.chapter_title, passage.part_number) == (3, "Arya", 1)  # chapter fields copied
    assert passages[0].character_start == 5 and passages[-1].character_end == chapter.character_end  # the whole chapter is covered
    assert sum(passage.token_count for passage in passages) / len(passages) > target_tokens * 0.6  # passages are filled, not tiny


def test_next_passage_starts_with_the_last_sentence_of_the_previous_one():  # one-sentence overlap
    chapter_text = make_chapter_text(300)  # chapter text
    passages = chunk_chapter(chapter_text, ChapterRecord(1, 1, "Jon", 0, len(chapter_text)), settings_for(200, 230))  # code under test
    for previous, following in zip(passages, passages[1:]):  # consecutive passages
        previous_sentences = [(previous.character_start + start, previous.character_start + end) for start, end in split_sentence_spans(previous.text)]  # its sentences
        assert following.character_start == previous_sentences[-1][0]  # starts at the previous passage's last sentence
        assert following.character_start < previous.character_end  # so they overlap


def test_overlap_is_skipped_when_switched_off_or_when_the_sentence_is_too_long_to_repeat():  # overlap rule
    chapter_text = make_chapter_text(200)  # chapter text
    chapter = ChapterRecord(1, 1, "Jon", 0, len(chapter_text))  # whole text
    for settings in (settings_for(200, 230, overlap_sentences=0), settings_for(200, 230, maximum_overlap_tokens=5)):  # no overlap either way
        passages = chunk_chapter(chapter_text, chapter, settings)  # code under test
        for previous, following in zip(passages, passages[1:]):  # consecutive passages
            assert following.character_start >= previous.character_end  # no shared text


def test_a_tiny_tail_is_merged_into_the_previous_passage_when_it_fits():  # minimum tail rule
    settings = settings_for(200, 230)  # default sizes
    for sentence_count in range(20, 60):  # many chapter lengths, so some end with a tiny tail
        chapter_text = make_chapter_text(sentence_count)  # chapter text
        passages = chunk_chapter(chapter_text, ChapterRecord(1, 1, "Jon", 0, len(chapter_text)), settings)  # code under test
        assert len(passages) == 1 or passages[-1].token_count >= settings.minimum_tail_tokens or passages[-2].token_count + passages[-1].token_count > settings.max_tokens  # a tail below the minimum is allowed only when merging would break the ceiling


def test_one_huge_sentence_is_cut_into_passages_under_the_ceiling():  # no sentence end anywhere
    chapter_text = " ".join("word%d" % number for number in range(1500))  # 1,500 words, no end mark
    passages = chunk_chapter(chapter_text, ChapterRecord(1, 1, "Jon", 0, len(chapter_text)), settings_for(200, 230))  # code under test
    assert len(passages) > 5 and all(passage.token_count <= 230 for passage in passages)  # all under the ceiling


def test_passages_never_cross_a_chapter_and_text_outside_chapters_becomes_chapterless_passages():  # chapters, a heading gap, an appendix tail
    first, second = make_chapter_text(60), make_chapter_text(60)  # two chapters
    heading_gap = " TYRION "  # a heading between them
    appendix = " APPENDIX " + make_chapter_text(80)  # text after the last chapter
    part_text = first + heading_gap + second + appendix  # the part
    second_start = len(first) + len(heading_gap)  # where chapter 2 begins
    chapters = [ChapterRecord(5, 1, "Arya", 0, len(first)), ChapterRecord(5, 2, "Tyrion", second_start, second_start + len(second))]  # two chapter spans
    passages = chunk_all_parts({5: part_text}, chapters, settings_for(200, 230))  # code under test
    for passage in passages:  # every passage lies in one stretch
        containing = [chapter for chapter in chapters if chapter.character_start <= passage.character_start and passage.character_end <= chapter.character_end]  # chapters holding it
        if passage.chapter_number is None:  # chapterless
            assert passage.character_start >= second_start + len(second)  # only the appendix tail
        else:  # chapter passage
            assert len(containing) == 1 and containing[0].chapter_number == passage.chapter_number  # inside exactly its chapter
    assert not any(passage.text.strip() in {"TYRION", "HEAD"} for passage in passages)  # the heading gap is not indexed
    assert len({passage.passage_id for passage in passages}) == len(passages)  # ids are unique
    assert all(passage.passage_id.startswith("bp-5-") for passage in passages)  # id carries the part
    assert any(passage.passage_id.startswith("bp-5-xxx-") for passage in passages)  # chapterless passages use xxx
    appendix = [passage for passage in passages if passage.chapter_number is None]  # the tail after the last chapter
    assert appendix and all(passage.chapter_title == "Appendix" for passage in appendix)  # public citations say "Appendix", not nothing
    assert format_chapter_label(appendix[0].chapter_number, appendix[0].chapter_title) == "Appendix"  # the label a visitor sees


def test_a_part_without_chapters_is_cut_as_one_chapterless_stretch():  # an unaligned book ships without chapter fields
    part_text = make_chapter_text(100)  # part text
    passages = chunk_all_parts({4: part_text}, [], settings_for(200, 230))  # code under test
    assert passages and all(passage.chapter_number is None and passage.chapter_title is None for passage in passages)  # book only
    assert passages[0].passage_id == "bp-4-xxx-000"  # numbering starts at 0
