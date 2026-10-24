# Checks that need the real books and pairs (run on the Mac with: python -m pytest -m data). They never print book text.
import pytest  # marker
from retrieval_evaluation.evaluation_settings import EvaluationSettings  # paths
from retrieval_evaluation.evaluation_stages.golden_sets_stage import build_golden_sets  # the whole golden build, including the FD checks

pytestmark = pytest.mark.data  # skipped unless asked for


def test_golden_sets_on_the_real_data_have_the_designed_counts():  # DESIGN.md §2 fact 14 and §14.3
    golden = build_golden_sets(EvaluationSettings())  # raises when a fact-development check fails (overlap, pair duplicate, evidence offset)
    counts = golden.counts  # the numbers
    assert counts["held_out_answerable"] == 816 and counts["held_out_located"] == 443  # the locator on the held-out stretch
    assert counts["held_out_development"] == 189 and counts["held_out_test"] == 196  # block split with the edge drop
    assert counts["fact_questions_word_for_word_in_pairs"] == 3 and counts["fact_development_questions_checked"] == len(golden.fact_development)  # checks ran
