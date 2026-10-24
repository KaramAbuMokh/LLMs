# A16: a 26-word copy of ANY passage (packed or not) is flagged; so are two book snippets and a snippet over 25 words. Invented text only.
from retrieval_evaluation.metrics.copyright_checks import build_passage_run_hashes, count_copied_runs, find_copyright_problems

UNPACKED_PASSAGE = " ".join("word%d" % number for number in range(40))  # 40 invented words: a passage that was never packed
PACKED_PASSAGE = " ".join("other%d" % number for number in range(40))  # the passage that was packed
RUN_HASHES = build_passage_run_hashes([UNPACKED_PASSAGE, PACKED_PASSAGE])  # runs of ALL passages
SNIPPET_SOURCE = {"source_type": "book_passage", "book": "B", "chapter": None, "snippet": "word0 word1 word2"}  # a short book snippet


def test_copy_of_an_unpacked_passage_is_flagged():  # the point of checking against all passages
    copy = " ".join("word%d" % number for number in range(5, 31))  # exactly 26 consecutive words
    assert count_copied_runs(copy, RUN_HASHES) == 1  # one window
    assert find_copyright_problems(copy, [], RUN_HASHES) != []  # reported
    assert count_copied_runs(" ".join("word%d" % number for number in range(5, 30)), RUN_HASHES) == 0  # 25 words are allowed


def test_two_book_snippets_in_one_response_are_flagged():  # one snippet per answer
    sources = [SNIPPET_SOURCE, dict(SNIPPET_SOURCE, snippet="word10 word11")]  # two book snippets
    assert any("2 book snippets" in problem for problem in find_copyright_problems("fine", sources, RUN_HASHES))  # flagged
    assert find_copyright_problems("fine", [SNIPPET_SOURCE], RUN_HASHES) == []  # one is fine


def test_long_snippet_and_pair_sources():  # 25 words at most; pair snippets are not book text
    long_snippet = dict(SNIPPET_SOURCE, snippet=" ".join("w%d" % number for number in range(26)))  # 26 words
    assert any("26 words" in problem for problem in find_copyright_problems("fine", [long_snippet], RUN_HASHES))  # flagged
    pairs = [{"source_type": "question_answer_pair", "snippet": "a question"}] * 3  # pair sources never count as book snippets
    assert find_copyright_problems("fine", pairs, RUN_HASHES) == []  # fine
