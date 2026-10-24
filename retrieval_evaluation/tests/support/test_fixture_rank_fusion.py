# The fixture's rank-fusion example and recorded hybrid orders, recomputed by a small reference fusion, so the fake retriever the other tests rely on is trustworthy
import pytest  # approximate comparisons
from retrieval_contracts.shared_fixture.fixture_files import load_fixture_expectations, load_fixture_golden_questions  # fixture
from retrieval_evaluation.tests.support.fixture_retrieval_fakes import FakeSearchSettings, FixtureRetriever  # fake retriever


def fuse_by_reciprocal_rank(ranked_id_lists, fusion_constant):  # reference fusion: 1-based ranks, ties by id ascending
    scores = {}  # id -> fused score
    for ranked_ids in ranked_id_lists:  # every list
        for rank, item_id in enumerate(ranked_ids, start=1):  # 1-based rank
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (fusion_constant + rank)  # reciprocal rank share
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))  # best first, ties by id


def test_the_recorded_example_is_reproduced():  # DESIGN.md §14.3
    example = load_fixture_expectations()["rank_fusion_example"]  # lists, constant and fused result
    fused = fuse_by_reciprocal_rank(example["lists"], example["constant"])  # recomputed
    assert [item_id for item_id, _ in fused] == [item_id for item_id, _ in example["fused"]]  # same order
    assert [score for _, score in fused] == pytest.approx([score for _, score in example["fused"]], abs=1e-8)  # same scores


def test_every_recorded_hybrid_ranking_is_the_fusion_of_its_dense_and_lexical_lists():  # and the fake retriever returns it
    expectations = load_fixture_expectations()["questions"]  # six questions
    retriever = FixtureRetriever(FakeSearchSettings())  # hybrid mode
    for golden_question in load_fixture_golden_questions():  # every fixture question
        recorded = expectations[golden_question.question_id]  # its recorded rankings
        fused = fuse_by_reciprocal_rank([[pid for pid, _ in recorded["passage_dense_ranking"]], [pid for pid, _ in recorded["passage_lexical_ranking"]]], 60)  # reference
        assert [pid for pid, _ in fused] == [pid for pid, _ in recorded["passage_fused_ranking"]]  # recorded order is the fusion
        returned = [candidate.record.passage_id for candidate in retriever.retrieve(golden_question.question).passage_candidates]  # fake retriever's order
        assert returned == [pid for pid, _ in fused]  # the fake serves exactly that order
