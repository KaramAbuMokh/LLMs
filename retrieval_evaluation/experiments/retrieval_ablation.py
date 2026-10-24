# Retrieval ablation on the G1 questions: every variant (search mode, fusion constant, reranker, chunk size) is measured with the same metrics
# on the passages the retriever ranks (DESIGN.md §11.3). Runs against objects passed in; imports nothing from the service at run time.
import dataclasses  # settings variants via replace
from dataclasses import dataclass  # frozen records
from typing import TYPE_CHECKING, Callable, Mapping, Sequence  # inputs
from retrieval_contracts.bundle_records import PassageRecord  # a stored passage
from retrieval_evaluation.evaluation_settings import RetrievalMeasureSettings  # ablation numbers
from retrieval_evaluation.golden_sets.golden_question_records import HeldOutQuestion  # G1 question
from retrieval_evaluation.metrics import ranking_metrics  # pure ranking measures
from retrieval_evaluation.metrics.passage_relevance import count_relevant_passages, flag_relevant_passages  # gold hits -> relevance

if TYPE_CHECKING:  # service types are named for readers and type checkers only
    from gpt2_demo_service.index_search.evidence_retriever import EvidenceRetriever
    from gpt2_demo_service.index_search.evidence_search_settings import EvidenceSearchSettings  # named for type checkers only

SHIPPED_BUNDLE_NAME = "shipped"  # the bundle built with the default chunk size


@dataclass(frozen=True)
class RetrievalVariant:  # one setting of the retrieval to measure
    name: str  # e.g. "dense_only"
    bundle_name: str  # which bundle (chunk size) it runs on
    settings: "EvidenceSearchSettings"  # the full search settings of this variant


@dataclass(frozen=True)
class VariantMeasurement:  # what one variant achieved on a question set
    name: str  # variant name
    bundle_name: str  # bundle name
    question_count: int  # questions measured
    metrics: Mapping[str, float]  # hit_at_<cutoff>, mean_reciprocal_rank_at_<depth>, packed-budget measures, ...
    mean_stage_milliseconds: Mapping[str, float]  # mean time per retrieval stage on this machine
    p95_total_milliseconds: float  # 95th percentile of the summed stage times
    first_relevant_ranks: Mapping[str, int | None]  # question id -> 1-based rank of the first relevant passage, None when absent


def build_search_variants(shipped_settings: "EvidenceSearchSettings", measure: RetrievalMeasureSettings) -> list[RetrievalVariant]:
    """The shipped bundle's variants: three search modes, two extra fusion constants and the reranker on top of hybrid."""
    hybrid = dataclasses.replace(shipped_settings, passage_search_mode="hybrid", reranker_enabled=False)  # the default arm
    variants = [RetrievalVariant("hybrid", SHIPPED_BUNDLE_NAME, hybrid)]  # reference first
    for mode in ("dense_only", "lexical_only"):  # the single searches
        variants.append(RetrievalVariant(mode, SHIPPED_BUNDLE_NAME, dataclasses.replace(hybrid, passage_search_mode=mode)))  # one change
    for constant in measure.fusion_constants:  # extra fusion constants
        variants.append(RetrievalVariant("hybrid_fusion_%d" % constant, SHIPPED_BUNDLE_NAME, dataclasses.replace(hybrid, fusion_constant=constant)))  # one change
    variants.append(RetrievalVariant("hybrid_rerank", SHIPPED_BUNDLE_NAME, dataclasses.replace(hybrid, reranker_enabled=True)))  # reranker on
    return variants  # six variants


def measure_variant(variant: RetrievalVariant, retriever: "EvidenceRetriever", questions: Sequence[HeldOutQuestion],  # one variant over a question set
                    index_passages: Sequence[PassageRecord], measure: RetrievalMeasureSettings) -> VariantMeasurement:
    per_question: dict[str, list[float]] = {}  # metric name -> value per question
    stage_totals: dict[str, list[float]] = {}  # stage name -> milliseconds per question
    ranks: dict[str, int | None] = {}  # question id -> first relevant rank
    totals: list[float] = []  # summed stage time per question
    depth, max_passages = measure.ranking_depth, variant.settings.max_passages  # depth of the deep measures, passages a prompt can hold
    for question in questions:  # every question once
        evidence = retriever.retrieve(question.question)  # ranked candidates and timings
        records = [candidate.record for candidate in evidence.passage_candidates]  # ranked passage records
        relevance = flag_relevant_passages(records, question.gold_hits)  # relevant flags, rank 1 first
        relevant_total = count_relevant_passages(index_passages, question.gold_hits)  # relevant passages in the whole index
        token_counts = [record.token_count for record in records]  # prompt cost of each
        ranks[question.question_id] = next((rank for rank, flag in enumerate(relevance, start=1) if flag), None)  # first relevant rank
        values = {"mean_reciprocal_rank_at_%d" % depth: ranking_metrics.reciprocal_rank_at_cutoff(relevance, depth),  # deep measures
                  "normalized_discounted_cumulative_gain_at_%d" % depth: ranking_metrics.normalized_discounted_cumulative_gain(relevance, depth, relevant_total),
                  "precision_at_%d" % depth: ranking_metrics.precision_at_cutoff(relevance, depth),
                  "recall_at_%d" % depth: ranking_metrics.recall_at_cutoff(relevance, depth, relevant_total),
                  "questions_without_relevant_passage": 1.0 if relevant_total == 0 else 0.0}
        values.update({"hit_at_%d" % cutoff: ranking_metrics.hit_at_cutoff(relevance, cutoff) for cutoff in measure.rank_cutoffs})  # shallow hits
        for budget in measure.packed_budgets:  # the chunk-size measures
            values["hit_within_packed_budget_%d" % budget] = ranking_metrics.hit_within_packed_budget(relevance, token_counts, budget, max_passages)  # hit
            precision = ranking_metrics.context_precision(relevance, token_counts, budget, max_passages)  # None when nothing fits
            values["context_precision_budget_%d" % budget] = 0.0 if precision is None else precision  # an empty context is not precise
        for name, value in values.items():  # file the question's values
            per_question.setdefault(name, []).append(value)  # collect
        totals.append(sum(evidence.stage_milliseconds.values()))  # this question's retrieval time
        for stage, milliseconds in evidence.stage_milliseconds.items():  # file the question's stage times
            stage_totals.setdefault(stage, []).append(milliseconds)  # collect
    return VariantMeasurement(  # the summary
        variant.name, variant.bundle_name, len(questions), {name: ranking_metrics.average_values(values) for name, values in per_question.items()},
        {stage: ranking_metrics.average_values(values) for stage, values in stage_totals.items()}, ranking_metrics.percentile_value(totals, 95.0), ranks,
    )


def run_retrieval_ablation(questions: Sequence[HeldOutQuestion], variants: Sequence[RetrievalVariant],  # measures all variants
                           retriever_for_variant: Callable[[RetrievalVariant], "EvidenceRetriever"],
                           passages_by_bundle: Mapping[str, Sequence[PassageRecord]], measure: RetrievalMeasureSettings) -> list[VariantMeasurement]:
    return [  # one measurement per variant, in the given order
        measure_variant(variant, retriever_for_variant(variant), questions, passages_by_bundle[variant.bundle_name], measure)  # build, then measure
        for variant in variants
    ]


def compare_reranker_effect(before: VariantMeasurement, after: VariantMeasurement, rank_cutoff: int) -> dict[str, list[str]]:
    """Question ids the reranker moved into the first rank_cutoff results ('fixed') or out of them ('broken')."""
    inside = lambda rank: rank is not None and rank <= rank_cutoff  # True when the first relevant passage is within the cutoff
    fixed = [question_id for question_id, rank in after.first_relevant_ranks.items() if inside(rank) and not inside(before.first_relevant_ranks[question_id])]
    broken = [question_id for question_id, rank in after.first_relevant_ranks.items() if not inside(rank) and inside(before.first_relevant_ranks[question_id])]  # moved out of the first results
    return {"fixed": fixed, "broken": broken}  # ids only
