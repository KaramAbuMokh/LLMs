# Tests of the book-run index: runs of 26 words across all passages, neighbours that share a sentence form one stretch, gaps and parts break it
from gpt2_demo_service.grounded_answers.book_run_index import FORBIDDEN_RUN_WORDS, build_book_run_index, join_adjacent_passages, list_words  # code under test
from retrieval_contracts.bundle_records import PassageRecord  # stored passage record


def make_record(text: str, character_start: int, part_number: int = 0) -> PassageRecord:  # an invented passage placed at an offset of its part text
    return PassageRecord("bp-%d-%d" % (part_number, character_start), part_number, "The Glass Coast", 1, 1, "Mira", character_start, character_start + len(text), 0, text)  # stored passage


def numbered_words(prefix: str, count: int) -> str:  # "prefix0 prefix1 ..." as one sentence
    return " ".join("%s%d" % (prefix, number) for number in range(count)) + "."  # distinct words, closed with a full stop


def test_a_run_of_26_words_is_found_and_25_are_not():  # the limit is "more than 25"
    passage = numbered_words("alpha", 40)  # 40 distinct words
    index = build_book_run_index([make_record(passage, 0)])  # one passage
    words = [word for word, _, _ in list_words(passage)]  # its words
    assert index.find_first_copied_word_index(words[:26]) == 25 and index.find_first_copied_word_index(words[:25]) is None  # the 26th word completes the first run
    assert index.find_first_copied_word_index(["unrelated"] + words[3:40]) == 26  # a run found anywhere inside other text: it starts after the first word, so it ends at word 26


def test_words_no_book_contains_never_match():  # a window with an unknown word cannot be a copy
    passage = numbered_words("alpha", 40)  # indexed text
    index = build_book_run_index([make_record(passage, 0)])  # one passage
    words = [word for word, _, _ in list_words(passage)]  # its words
    words[10] = "intruder"  # one word replaced
    assert index.find_first_copied_word_index(words[:30]) is None  # every 26-word window of 30 words contains word 10


def test_neighbours_sharing_a_sentence_read_as_one_stretch():  # a run may start in one passage and end in the next
    shared = numbered_words("shared", 5)  # the sentence both passages hold
    first = numbered_words("first", 12) + " " + shared  # ends with the shared sentence
    second = shared + " " + numbered_words("second", 12)  # starts with it
    records = [make_record(first, 0), make_record(second, len(first) - len(shared))]  # the second starts where the shared sentence starts
    index = build_book_run_index(records)  # one stretch of 12 + 5 + 12 words
    crossing = [word for word, _, _ in list_words(first)][2:] + [word for word, _, _ in list_words(second)][5:12]  # 15 + 7 words around the join
    assert len(join_adjacent_passages(records)) == 1 and index.find_first_copied_word_index(crossing[:26]) is None  # 22 words only: below the limit
    longer = [word for word, _, _ in list_words(first)] + [word for word, _, _ in list_words(second)][5:]  # the whole stretch without the repeated sentence
    assert len(longer) == 29 and index.find_first_copied_word_index(longer) == 25  # 29 words found as one run: the shared sentence counted once


def test_a_gap_or_another_part_breaks_the_stretch():  # words far apart in the book never join
    first, second = numbered_words("alpha", 15), numbered_words("beta", 15)  # two 15-word passages
    assert len(join_adjacent_passages([make_record(first, 0), make_record(second, 1000)])) == 2  # a gap of 984 characters
    assert len(join_adjacent_passages([make_record(first, 0, part_number=0), make_record(second, len(first) + 1, part_number=1)])) == 2  # another part
    index = build_book_run_index([make_record(first, 0), make_record(second, 1000)])  # two stretches
    glued = [word for word, _, _ in list_words(first)] + [word for word, _, _ in list_words(second)]  # 30 words glued together
    assert index.find_first_copied_word_index(glued) is None  # no run of 26


def test_the_index_is_deterministic_and_counts_distinct_runs():  # same books, same index
    records = [make_record(numbered_words("alpha", 40), 0), make_record(numbered_words("beta", 30), 1000)]  # 15 + 5 windows
    first, second = build_book_run_index(records), build_book_run_index(records)  # built twice
    assert first.run_count == 15 + 5 and (first.sorted_run_hashes == second.sorted_run_hashes).all()  # windows per stretch, identical hashes
    assert FORBIDDEN_RUN_WORDS == 26 and first.run_length == 26  # the policy number


def test_an_empty_index_matches_nothing():  # no passages at all
    index = build_book_run_index([])  # nothing to remember
    assert index.run_count == 0 and index.find_first_copied_word_index(["any"] * 40) is None  # nothing found
