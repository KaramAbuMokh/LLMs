# The `retrieval` stage: ablation on G1 development, the three decisions (search mode, reranker, chunk size), then ONE run of the chosen
# retrieval on G1 test. Writes results/retrieval_ablation.json.
import dataclasses  # settings variants and result dictionaries
from typing import Any
from retrieval_contracts.bundle_manifest import PASSAGE_RECORDS_FILE_NAME  # passage file name
from retrieval_contracts.bundle_records import PassageRecord, read_json_lines  # stored passages
from retrieval_evaluation.evaluation_settings import EvaluationSettings  # numbers
from retrieval_evaluation.experiments.evaluation_report_writing import write_json_results  # result files
from retrieval_evaluation.experiments.retrieval_ablation import (  # variants and measurements
    SHIPPED_BUNDLE_NAME, RetrievalVariant, VariantMeasurement, build_search_variants, compare_reranker_effect, measure_variant, run_retrieval_ablation,
)
from retrieval_evaluation.experiments.settings_recommendation import choose_chunk_size_name, choose_search_mode, decide_reranker  # decision rules
from retrieval_evaluation.evaluation_stages.evaluation_runtime import EvaluationRuntime  # service objects
from retrieval_evaluation.golden_sets.golden_question_records import GoldenSets  # question sets

ABLATION_FILE_NAME = "retrieval_ablation.json"  # inside the results folder


def build_chosen_search_settings(shipped_settings: Any, decisions: dict[str, Any]) -> Any:  # the shipped settings with the chosen mode and reranker switch
    return dataclasses.replace(shipped_settings, passage_search_mode=decisions["passage_search_mode"], reranker_enabled=decisions["reranker_enabled"])  # two changes


def run_retrieval_stage(golden: GoldenSets, runtime: EvaluationRuntime, settings: EvaluationSettings) -> dict[str, Any]:  # ablation, three decisions, one test run
    measure = settings.retrieval  # ablation numbers
    passages_by_bundle = {name: read_json_lines(folder / PASSAGE_RECORDS_FILE_NAME, PassageRecord) for name, folder in runtime.bundle_folders.items()}  # per bundle
    variants = build_search_variants(runtime.shipped_search_settings, measure)  # shipped-bundle variants
    hybrid_settings = variants[0].settings  # the hybrid reference
    chunk_variants = [RetrievalVariant("hybrid_" + name, name, hybrid_settings) for name in runtime.bundle_folders if name != SHIPPED_BUNDLE_NAME]  # other sizes
    run_retriever = lambda variant: runtime.build_retriever(variant.bundle_name, variant.settings)  # one retriever per variant
    measurements = run_retrieval_ablation(golden.held_out_development, variants + chunk_variants, run_retriever, passages_by_bundle, measure)  # development only
    by_name: dict[str, VariantMeasurement] = {measurement.name: measurement for measurement in measurements}  # lookup
    reranker_enabled, reranker_reason = decide_reranker(  # rule 2
        by_name["hybrid"].metrics["hit_at_3"], by_name["hybrid_rerank"].metrics["hit_at_3"], by_name["hybrid_rerank"].p95_total_milliseconds, measure)
    chunk_measurements = [by_name["hybrid"]] + [by_name["hybrid_" + variant.bundle_name] for variant in chunk_variants]  # one hybrid arm per chunk size
    packed_key = "hit_within_packed_budget_%d" % measure.chunk_comparison_budget  # the only measure compared across sizes
    decisions = {  # the three retrieval choices
        "passage_search_mode": choose_search_mode({measurement.name: measurement.metrics["hit_at_5"] for measurement in measurements}, measure),  # rule 1
        "reranker_enabled": reranker_enabled, "reranker_reason": reranker_reason,
        "chunk_size_name": choose_chunk_size_name({measurement.bundle_name: measurement.metrics[packed_key] for measurement in chunk_measurements}, SHIPPED_BUNDLE_NAME, measure),
        "fusion_constant": runtime.shipped_search_settings.fusion_constant,  # kept as shipped: the other constants are reported only
    }
    chosen = RetrievalVariant("chosen", decisions["chunk_size_name"], build_chosen_search_settings(runtime.shipped_search_settings, decisions))  # final arm
    test_run = measure_variant(chosen, run_retriever(chosen), golden.held_out_test, passages_by_bundle[chosen.bundle_name], measure)  # ONE run on test
    result = {"variants": [dataclasses.asdict(measurement) for measurement in measurements if measurement.bundle_name == SHIPPED_BUNDLE_NAME],  # shipped-bundle variants
              "chunk_sizes": [dataclasses.asdict(measurement) for measurement in chunk_measurements], "decisions": decisions,
              "reranker_effect": compare_reranker_effect(by_name["hybrid"], by_name["hybrid_rerank"], 3), "test_run": dataclasses.asdict(test_run)}
    write_json_results(settings.locations.results_folder / ABLATION_FILE_NAME, result)  # record
    return result  # for the next stage
