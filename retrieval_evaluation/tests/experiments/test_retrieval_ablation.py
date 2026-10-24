# Retrieval ablation on the fixture's recorded rankings: hand-computed numbers, variants as settings replacements, the reranker comparison
import dataclasses  # copy a record
import pytest  # approximate comparisons
from retrieval_contracts.bundle_manifest import PASSAGE_RECORDS_FILE_NAME  # fixture passage file
from retrieval_contracts.bundle_records import PassageRecord, read_json_lines  # stored passages
from retrieval_contracts.rare_shingle_location import ShingleHit  # gold hits
from retrieval_contracts.shared_fixture.fixture_files import FIXTURE_BUNDLE_FOLDER  # fixture
from retrieval_evaluation.evaluation_settings import RetrievalMeasureSettings  # numbers
from retrieval_evaluation.experiments.retrieval_ablation import (  # under test
    RetrievalVariant, VariantMeasurement, build_search_variants, compare_reranker_effect, measure_variant, run_retrieval_ablation,
)
from retrieval_evaluation.tests.support.fixture_retrieval_fakes import FakeSearchSettings, FixtureRetriever  # fakes
from retrieval_evaluation.tests.support.fixture_stage_inputs import build_fixture_held_out_questions  # G1-like questions

PASSAGES = read_json_lines(FIXTURE_BUNDLE_FOLDER / PASSAGE_RECORDS_FILE_NAME, PassageRecord)  # the six fixture passages


def make_question_with_gold_in_second_passage():  # fg-3 with a gold hit in bp-1-001-001, which hybrid ranks second
    question = next(question for question in build_fixture_held_out_questions("development") if question.question_id.endswith("fg-3"))  # fg-3
    return dataclasses.replace(question, gold_hits=(ShingleHit(1, 140, 160),))  # a hit inside passage bp-1-001-001 (136-200) only


def test_measure_variant_matches_hand_computed_values():  # relevant passage at rank 2 under hybrid
    variant = RetrievalVariant("hybrid", "shipped", FakeSearchSettings())  # reference arm
    measure = RetrievalMeasureSettings(packed_budgets=(30, 300))  # budget 30 holds only the first passage (29 tokens)
    result = measure_variant(variant, FixtureRetriever(variant.settings), [make_question_with_gold_in_second_passage()], PASSAGES, measure)
    metrics = result.metrics  # shorthand
    assert (metrics["hit_at_1"], metrics["hit_at_3"], metrics["hit_at_10"]) == (0.0, 1.0, 1.0)  # rank 2
    assert metrics["mean_reciprocal_rank_at_10"] == 0.5 and metrics["recall_at_10"] == 1.0  # reciprocal rank and recall
    assert metrics["normalized_discounted_cumulative_gain_at_10"] == pytest.approx(0.630930, abs=1e-6)  # (1 / log2 3) / 1
    assert metrics["precision_at_10"] == pytest.approx(0.1)  # one relevant of the ten slots
    assert metrics["hit_within_packed_budget_30"] == 0.0 and metrics["hit_within_packed_budget_300"] == 1.0  # only the bigger budget reaches rank 2
    assert metrics["context_precision_budget_300"] == pytest.approx(1 / 3)  # three passages packed, one relevant
    assert result.first_relevant_ranks == {"g1-development-fg-3": 2} and result.p95_total_milliseconds == 3.0  # rank and encode + dense time


def test_variants_are_replacements_of_the_shipped_settings():  # one change each
    variants = build_search_variants(FakeSearchSettings(), RetrievalMeasureSettings())  # built
    by_name = {variant.name: variant.settings for variant in variants}  # lookup
    assert list(by_name) == ["hybrid", "dense_only", "lexical_only", "hybrid_fusion_20", "hybrid_fusion_100", "hybrid_rerank"]  # the ablation list
    assert by_name["dense_only"].passage_search_mode == "dense_only" and by_name["hybrid_fusion_20"].fusion_constant == 20  # changed fields
    assert by_name["hybrid_rerank"].reranker_enabled and not by_name["hybrid"].reranker_enabled  # reranker only in its own arm


def test_run_ablation_builds_one_retriever_per_variant_and_measures_the_reranker_time():  # runner glue
    variants = build_search_variants(FakeSearchSettings(), RetrievalMeasureSettings())  # six variants
    questions = build_fixture_held_out_questions("development")  # four questions
    built = []  # variants that got a retriever
    results = run_retrieval_ablation(questions, variants, lambda variant: built.append(variant.name) or FixtureRetriever(variant.settings), {"shipped": PASSAGES}, RetrievalMeasureSettings())
    assert built == [variant.name for variant in variants] and len(results) == 6  # one each, in order
    by_name = {result.name: result for result in results}  # lookup
    assert by_name["hybrid_rerank"].mean_stage_milliseconds["rerank"] == 10.0 and "rerank" not in by_name["hybrid"].mean_stage_milliseconds  # timed
    assert by_name["hybrid"].metrics["hit_at_1"] == 1.0 and by_name["hybrid"].question_count == 4  # every fixture answer ranks first


def test_reranker_effect_lists_fixed_and_broken_questions():  # who moved into or out of the first 3
    make = lambda ranks: VariantMeasurement("variant", "shipped", 4, {}, {}, 0.0, ranks)  # a measurement with given first ranks
    before, after = make({"moves_in": 5, "moves_out": 2, "never_found": None, "stays_first": 1}), make({"moves_in": 1, "moves_out": 4, "never_found": 3, "stays_first": 1})  # reranker: two in, one out
    assert compare_reranker_effect(before, after, 3) == {"fixed": ["moves_in", "never_found"], "broken": ["moves_out"]}  # ids only
