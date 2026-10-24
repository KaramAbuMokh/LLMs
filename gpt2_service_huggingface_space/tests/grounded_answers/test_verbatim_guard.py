# Tests of the copy guard: it cuts a long copy of ANY book passage at a sentence end, and leaves short overlaps and other text alone
from gpt2_demo_service.grounded_answers.book_run_index import build_book_run_index  # the memory of the books
from gpt2_demo_service.grounded_answers.verbatim_guard import limit_verbatim_overlap  # code under test
from retrieval_contracts.bundle_records import PassageRecord  # stored passage record
from retrieval_contracts.shared_fixture.fixture_files import load_fixture_part_texts  # the invented fixture books

YSOLDE_PASSAGE = (  # the 33-word invented fixture passage
    "Queen Ysolde ruled the Glass Coast from the harbor city of Lumen. She wore a crown of sea glass and never carried a sword. "
    "Her councillors feared the storms more than any army."
)


def make_index(*texts: str):  # an index over passages that sit far apart in the book
    return build_book_run_index([PassageRecord("bp-0-%d" % number, 0, "The Glass Coast", 1, 1, "Mira", number * 10000, number * 10000 + len(text), 0, text) for number, text in enumerate(texts)])  # one record per text


def test_the_fixture_passage_is_33_words_long():  # guards the test data
    assert YSOLDE_PASSAGE in load_fixture_part_texts()[1] and len(YSOLDE_PASSAGE.split()) == 33  # exact stretch of the invented book


def test_full_copy_is_cut_back_to_the_last_sentence_before_the_limit():  # a copy passes 25 words inside the third sentence
    guarded = limit_verbatim_overlap(YSOLDE_PASSAGE, make_index(YSOLDE_PASSAGE))  # the model copied the passage
    assert guarded.was_cut  # detected
    assert guarded.text == "Queen Ysolde ruled the Glass Coast from the harbor city of Lumen. She wore a crown of sea glass and never carried a sword."  # two sentences, 23 words
    assert len(guarded.text.split()) <= 25  # within the limit


def test_copy_without_a_sentence_end_is_cut_after_25_words_with_an_ellipsis():  # no punctuation to cut back to
    passage = " ".join("word%d" % number for number in range(40))  # 40 distinct words, no punctuation
    guarded = limit_verbatim_overlap(passage, make_index(passage))  # fully copied
    assert guarded.was_cut and guarded.text == " ".join("word%d" % number for number in range(25)) + "…"  # 25 words and the ellipsis


def test_exactly_25_shared_words_are_allowed():  # the limit is "more than 25"
    passage = " ".join("word%d" % number for number in range(25)) + "."  # 25 words
    guarded = limit_verbatim_overlap(passage, make_index(passage))  # fully copied
    assert not guarded.was_cut and guarded.text == passage  # unchanged


def test_unrelated_text_and_a_missing_index_pass_untouched():  # nothing to compare
    unrelated = "Winter is coming for everyone."  # not in the books
    assert not limit_verbatim_overlap(unrelated, make_index(YSOLDE_PASSAGE)).was_cut and limit_verbatim_overlap(unrelated, make_index(YSOLDE_PASSAGE)).text == unrelated  # not cut
    assert limit_verbatim_overlap(YSOLDE_PASSAGE, None).text == YSOLDE_PASSAGE and not limit_verbatim_overlap(YSOLDE_PASSAGE, None).was_cut  # no index, no check


def test_a_copy_of_a_passage_that_was_never_packed_is_cut_too():  # the index holds every passage, not only the three in the prompt
    other = " ".join("other%d" % number for number in range(30)) + "."  # a passage nobody retrieved
    guarded = limit_verbatim_overlap("Short intro. " + other, make_index(YSOLDE_PASSAGE, other))  # memorised from elsewhere in the books
    assert guarded.was_cut and guarded.text == "Short intro."  # cut back to the sentence before the copy


def test_a_run_never_crosses_from_one_passage_into_the_next():  # two passages far apart are separate texts
    first, second = " ".join("alpha%d" % number for number in range(15)), " ".join("beta%d" % number for number in range(15))  # two 15-word passages
    assert not limit_verbatim_overlap(first + " " + second, make_index(first, second)).was_cut  # two runs of 15, not one of 30


def test_case_and_curly_apostrophes_do_not_hide_a_copy():  # the comparison folds both
    passage = "The queen’s " + " ".join("word%d" % number for number in range(30))  # starts with a curly apostrophe
    copy = "the QUEEN'S " + " ".join("WORD%d" % number for number in range(30))  # same words, other case and apostrophe
    assert limit_verbatim_overlap(copy, make_index(passage)).was_cut  # still detected
