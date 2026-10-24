# Tests of the evidence retriever against the shared fixture: rankings, fusion, gate, pair examples, routes, modes, reranker
from dataclasses import replace  # copies settings with one change
import numpy  # score comparison
import pytest  # expected errors
from gpt2_demo_service.index_search.evidence_records import RETRIEVAL_STAGE_NAMES, EvidenceRoute, build_empty_evidence  # records under test
from gpt2_demo_service.index_search.evidence_retriever import EvidenceRetriever  # code under test
from gpt2_demo_service.index_search.rank_fusion import fuse_rankings  # builds the expected pair order
from retrieval_contracts.shared_fixture.fixture_encoders import HashingTextEncoder, TokenOverlapPairScorer  # fakes
from retrieval_contracts.shared_fixture.fixture_files import load_fixture_golden_questions  # fixture questions

SCORE_TOLERANCE = 2e-6  # the expectations are rounded to six decimals


@pytest.fixture(scope="module")
def retriever(fixture_bundle, fixture_search_settings) -> EvidenceRetriever:  # retriever over the fixture bundle with the fixture settings
    return EvidenceRetriever(fixture_bundle, HashingTextEncoder(), fixture_search_settings)  # no reranker


def test_candidates_equal_the_fixture_expectations(retriever, fixture_expectations):  # fused order, best dense score, pair candidates
    for question in load_fixture_golden_questions():  # fg-1 .. fg-6
        expected = fixture_expectations["questions"][question.question_id]  # recorded numbers
        evidence = retriever.retrieve(question.question)  # our retrieval
        assert [c.record.passage_id for c in evidence.passage_candidates] == [item[0] for item in expected["passage_fused_ranking"]]  # fused order
        assert numpy.allclose([c.fused_rank for c in evidence.passage_candidates], range(1, 7))  # 1-based fused ranks
        assert numpy.isclose(evidence.best_evidence_score, expected["best_passage_dense_score"], atol=SCORE_TOLERANCE)  # gate score
        pair_lists = [[item[0] for item in expected["pair_dense_ranking"]], [item[0] for item in expected["pair_lexical_ranking"]]]  # recorded pair rankings
        assert [p.record.pair_id for p in evidence.pair_candidates] == [c.item_id for c in fuse_rankings(pair_lists, 60)]  # fused pair order
        recorded_similarity = {item[0]: item[1] for item in expected["pair_dense_ranking"]}  # recorded cosines
        assert all(numpy.isclose(p.question_similarity, recorded_similarity[p.record.pair_id], atol=SCORE_TOLERANCE) for p in evidence.pair_candidates)  # exact cosines


def test_answerable_questions_get_passages_and_unanswerable_get_none(retriever):  # the gate separates fg-1..4 from fg-5/6
    routes = {question.question_id: retriever.retrieve(question.question).route for question in load_fixture_golden_questions()}  # route per question
    assert routes["fg-5"] == EvidenceRoute.NO_EVIDENCE and routes["fg-6"] == EvidenceRoute.NO_EVIDENCE  # nothing good enough
    assert all(routes[name] != EvidenceRoute.NO_EVIDENCE for name in ("fg-1", "fg-2", "fg-3", "fg-4"))  # evidence found


def test_passages_are_limited_and_best_first(retriever):  # at most max_passages after the gate
    evidence = retriever.retrieve("Where are the keys of the salt tower kept?")  # fg-3
    assert [p.record.passage_id for p in evidence.passages] == ["bp-1-001-000", "bp-1-001-001", "bp-1-002-000"]  # top three
    assert len(evidence.passage_candidates) == 6  # candidates are not gated


def test_pair_examples_need_the_threshold(retriever):  # qa-103 reaches 0.5 for fg-3, nothing reaches 0.45 for fg-4
    assert [p.record.pair_id for p in retriever.retrieve("Where are the keys of the salt tower kept?").pair_examples] == ["qa-103"]  # one example
    assert retriever.retrieve("Who is the queen's chief counsellor?").pair_examples == ()  # no example


def test_pair_search_is_skipped_when_pairs_are_off(fixture_bundle, fixture_search_settings):  # the explicit flag, not a magic threshold
    pairs_off = EvidenceRetriever(fixture_bundle, HashingTextEncoder(), replace(fixture_search_settings, pair_examples_enabled=False))  # same numbers, pairs switched off
    pairs_on = EvidenceRetriever(fixture_bundle, HashingTextEncoder(), fixture_search_settings)  # pairs on
    question = "Where are the keys of the salt tower kept?"  # fg-3 finds qa-103 when pairs are on
    evidence = pairs_off.retrieve(question)  # no pair work at all
    assert evidence.pair_candidates == () and evidence.pair_examples == ()  # nothing searched, nothing shown
    assert [p.record.passage_id for p in evidence.passages] == [p.record.passage_id for p in pairs_on.retrieve(question).passages]  # the passages are untouched
    assert pairs_on.retrieve(question).pair_examples and evidence.route == EvidenceRoute.PASSAGES_ONLY  # while the same question shows an example when pairs are on


def test_stage_names_are_known(retriever):  # only documented stage names appear, rerank is absent when the reranker is off
    stages = retriever.retrieve("Who forged the sword called Dawnlight?").stage_milliseconds  # timings
    assert set(stages) <= set(RETRIEVAL_STAGE_NAMES) and "rerank" not in stages and "encode" in stages  # documented names only
    assert all(milliseconds >= 0.0 for milliseconds in stages.values())  # sane numbers


def test_single_search_modes_keep_the_exact_cosines(fixture_bundle, fixture_search_settings):  # the gate score is mode-independent
    hybrid = EvidenceRetriever(fixture_bundle, HashingTextEncoder(), fixture_search_settings).retrieve("Who forged the sword called Dawnlight?")  # reference
    for mode in ("dense_only", "lexical_only"):  # the two single searches
        evidence = EvidenceRetriever(fixture_bundle, HashingTextEncoder(), replace(fixture_search_settings, passage_search_mode=mode)).retrieve("Who forged the sword called Dawnlight?")  # retriever in one single-search mode
        assert numpy.isclose(evidence.best_evidence_score, hybrid.best_evidence_score)  # same gate score
    lexical_only = EvidenceRetriever(fixture_bundle, HashingTextEncoder(), replace(fixture_search_settings, passage_search_mode="lexical_only")).retrieve("Who forged the sword called Dawnlight?")  # retriever that uses the lexical search only
    assert [c.record.passage_id for c in lexical_only.passage_candidates] == ["bp-0-001-000", "bp-1-002-000"]  # only lexical hits are candidates
    assert "dense_search" in lexical_only.stage_milliseconds  # the pair search still ran densely


def test_reranker_reorders_the_head_and_gates_on_its_scores(fixture_bundle, fixture_search_settings, fixture_expectations):  # overlap scorer as stand-in
    settings = replace(fixture_search_settings, reranker_enabled=True, rerank_candidate_count=6, rerank_evidence_floor=1.0)  # reranker on
    retriever = EvidenceRetriever(fixture_bundle, HashingTextEncoder(), settings, TokenOverlapPairScorer())  # with the stand-in scorer
    for question in load_fixture_golden_questions():  # fg-1 .. fg-6
        recorded = fixture_expectations["questions"][question.question_id]["passage_overlap_rerank_scores"]  # score per passage id
        evidence = retriever.retrieve(question.question)  # our retrieval
        scores = [candidate.rerank_score for candidate in evidence.passage_candidates]  # rerank scores in final order
        assert all(candidate.rerank_score == recorded[candidate.record.passage_id] for candidate in evidence.passage_candidates)  # recorded scores
        assert scores == sorted(scores, reverse=True) and "rerank" in evidence.stage_milliseconds  # best first, stage timed
        assert evidence.best_evidence_score == max(scores)  # the gate reads the rerank scores
        assert bool(evidence.passages) == (max(scores) >= 1.0)  # passages only at or above the rerank floor


def test_reranker_only_reads_the_head(fixture_bundle, fixture_search_settings):  # candidates behind the head keep their fused order and no score
    settings = replace(fixture_search_settings, reranker_enabled=True, rerank_candidate_count=2)  # only two are read
    evidence = EvidenceRetriever(fixture_bundle, HashingTextEncoder(), settings, TokenOverlapPairScorer()).retrieve("Where are the keys of the salt tower kept?")  # fg-3
    assert [c.rerank_score is not None for c in evidence.passage_candidates] == [True, True, False, False, False, False]  # head scored only
    assert [c.fused_rank for c in evidence.passage_candidates[2:]] == [3, 4, 5, 6]  # the rest keeps fused order


def test_reranker_without_scorer_is_refused(fixture_bundle, fixture_search_settings):  # a wiring bug is caught at startup
    with pytest.raises(ValueError, match="pair scorer"):  # named reason
        EvidenceRetriever(fixture_bundle, HashingTextEncoder(), replace(fixture_search_settings, reranker_enabled=True))  # no scorer


def test_empty_evidence_for_off_and_failed():  # nothing found, no stages
    for route in (EvidenceRoute.RETRIEVAL_OFF, EvidenceRoute.RETRIEVAL_FAILED):  # the two routes without a search
        evidence = build_empty_evidence(route)  # empty result
        assert evidence.route == route and evidence.passages == () and evidence.pair_examples == () and evidence.stage_milliseconds == {}  # all empty
        assert evidence.best_evidence_score is None  # no gate score
