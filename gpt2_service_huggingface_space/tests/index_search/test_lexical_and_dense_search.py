# Tests of the exact searches: lexical and dense rankings equal the fixture expectations, ties go to the lower row, edge cases are quiet
import numpy  # arrays
from gpt2_demo_service.index_search.dense_search import find_top_dense_matches  # code under test
from gpt2_demo_service.index_search.lexical_search import find_top_lexical_matches, score_lexical_matches  # code under test
from retrieval_contracts.shared_fixture.fixture_encoders import HashingTextEncoder  # fake encoder
from retrieval_contracts.shared_fixture.fixture_files import load_fixture_golden_questions  # fixture questions

SCORE_TOLERANCE = 2e-6  # the expectations are rounded to six decimals


def test_lexical_ranking_matches_the_fixture(fixture_bundle, fixture_expectations):  # passage and pair BM25 rankings of all six questions
    for question in load_fixture_golden_questions():  # fg-1 .. fg-6
        expected = fixture_expectations["questions"][question.question_id]  # recorded rankings
        passage_ranking = find_top_lexical_matches(fixture_bundle.passage_lexical, question.question, 6)  # our ranking
        assert [fixture_bundle.passages[ranked.row].passage_id for ranked in passage_ranking] == [item[0] for item in expected["passage_lexical_ranking"]]  # order
        assert numpy.allclose([ranked.score for ranked in passage_ranking], [item[1] for item in expected["passage_lexical_ranking"]], atol=SCORE_TOLERANCE)  # scores
        pair_ranking = find_top_lexical_matches(fixture_bundle.pair_lexical, question.question, 6)  # pair ranking
        assert [fixture_bundle.pairs[ranked.row].pair_id for ranked in pair_ranking] == [item[0] for item in expected["pair_lexical_ranking"]]  # order


def test_dense_ranking_matches_the_fixture(fixture_bundle, fixture_expectations):  # cosine rankings with the instruction prefix for passages
    encoder = HashingTextEncoder()  # the encoder that built the fixture vectors
    instruction = encoder.fingerprint.passage_query_instruction  # prefix of passage questions
    for question in load_fixture_golden_questions():  # fg-1 .. fg-6
        expected = fixture_expectations["questions"][question.question_id]  # recorded rankings
        passage_query = encoder.encode_texts([instruction + question.question])[0]  # passage-search query
        ranking = find_top_dense_matches(fixture_bundle.passage_vectors, passage_query, 6)  # all passages
        assert [fixture_bundle.passages[ranked.row].passage_id for ranked in ranking] == [item[0] for item in expected["passage_dense_ranking"]]  # order incl. ties
        assert numpy.allclose([ranked.score for ranked in ranking], [item[1] for item in expected["passage_dense_ranking"]], atol=SCORE_TOLERANCE)  # scores
        question_query = encoder.encode_texts([question.question])[0]  # question-to-question query (no prefix)
        pair_ranking = find_top_dense_matches(fixture_bundle.question_vectors, question_query, 4)  # all pairs
        assert [fixture_bundle.pairs[ranked.row].pair_id for ranked in pair_ranking] == [item[0] for item in expected["pair_dense_ranking"]]  # order


def test_question_without_known_words_matches_nothing(fixture_bundle):  # "???" has no lexical tokens
    assert find_top_lexical_matches(fixture_bundle.passage_lexical, "???", 5) == []  # empty ranking
    assert not score_lexical_matches(fixture_bundle.passage_lexical, "???").any()  # all-zero scores of the right length


def test_dense_top_k_equals_a_full_sort_with_ties_by_row():  # random data with many ties at the border
    random_generator = numpy.random.default_rng(3)  # repeatable data
    vectors = random_generator.integers(0, 3, size=(200, 4)).astype(numpy.float32)  # small integers make many equal scores
    query = numpy.array([1.0, 0.5, 0.0, 0.25], dtype=numpy.float32)  # a query
    scores = vectors @ query  # reference scores
    expected_rows = sorted(range(200), key=lambda row: (-scores[row], row))[:17]  # full sort, ties by row
    assert [ranked.row for ranked in find_top_dense_matches(vectors, query, 17)] == expected_rows  # same top 17
    assert find_top_dense_matches(vectors, query, 0) == []  # a zero limit returns nothing
    assert len(find_top_dense_matches(vectors, query, 500)) == 200  # a limit above the corpus returns everything
