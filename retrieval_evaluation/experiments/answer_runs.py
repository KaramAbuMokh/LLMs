# Answer runs (DESIGN.md §11.5): retrieval runs ONCE per question, then the service's own answer_from_evidence is called once per configuration
# and sample with a seeded generator, so seeds are identical across configurations. No route, guard or source logic is rebuilt here.
import time  # generation timing
from dataclasses import dataclass  # frozen tool bundle
from typing import TYPE_CHECKING, Callable, Mapping, Sequence  # inputs
from retrieval_evaluation.experiments.answer_run_planning import AnswerConfiguration, AnswerItem, GenerationClock  # plan records
from retrieval_evaluation.metrics.answer_overlap_scoring import find_unsupported_names  # invented-name detector
from retrieval_evaluation.metrics.answer_run_scoring import AnswerRecord  # one generated answer

if TYPE_CHECKING:  # service types, for readers and type checkers only
    from gpt2_demo_service.grounded_answers.grounded_answer_service import AnswerRequest, GroundedAnswer, GroundedAnswerService
    from gpt2_demo_service.index_search.evidence_retriever import EvidenceRetriever  # named for type checkers only


@dataclass(frozen=True)
class AnswerRunTools:  # the service objects and helpers a run needs, wired by run_evaluation_command.py
    retriever: "EvidenceRetriever"  # searches the bundle
    answer_service: "GroundedAnswerService"  # packs, generates, guards and cites
    build_answer_request: Callable[[str], "AnswerRequest"]  # question -> request with the run's sampling and stop settings
    make_random_generator: Callable[[int], object]  # seed -> a seeded torch generator
    find_supporting_texts: Callable[[Sequence[str], Sequence[str]], list[str]]  # (passage ids, pair ids) -> their texts
    find_copyright_problems: Callable[[str, Sequence[Mapping[str, object]]], list[str]]  # (completion, sources) -> A16 findings


def describe_sources(answer: "GroundedAnswer") -> list[dict[str, object]]:  # the public sources as plain dictionaries
    return [{"source_type": str(source.source_type), "book": source.book, "chapter": source.chapter, "snippet": source.snippet} for source in answer.sources]


def build_answer_record(item: AnswerItem, configuration: AnswerConfiguration, sample: int, answer: "GroundedAnswer", tools: AnswerRunTools) -> AnswerRecord:  # one answer as a stored record
    sources = describe_sources(answer)  # what a visitor would see
    supporting_texts = [item.question] + tools.find_supporting_texts(answer.passage_ids, answer.pair_ids)  # question plus the packed context
    return AnswerRecord(  # ids, scores, counts and the model's own words: no passage text
        item.question_id, item.set_name, configuration.name, sample, answer.completion, answer.retrieval_used, str(answer.route),
        tuple(answer.passage_ids), tuple(answer.pair_ids), answer.best_evidence_score, answer.prompt_token_count, answer.answer_token_count,
        str(answer.stop_reason), answer.verbatim_guard_cut, sum(1 for source in sources if source["source_type"] == "book_passage" and source["snippet"]),
        tuple(find_unsupported_names(answer.completion, supporting_texts)), tuple(tools.find_copyright_problems(answer.completion, sources)),
    )


def run_answer_items(items: Sequence[AnswerItem], configurations: Sequence[AnswerConfiguration], tools: AnswerRunTools, clock: GenerationClock,  # generates every item under every configuration
                     seed_multiplier: int, report_progress: Callable[[str], None] = lambda message: None) -> list[AnswerRecord]:
    records = []  # every generated answer
    for position, item in enumerate(items, start=1):  # question by question
        evidence = tools.retriever.retrieve(item.question)  # retrieval runs once, every configuration reuses it
        answer_request = tools.build_answer_request(item.question)  # same request for every configuration
        for configuration in configurations:  # every layout and budget
            for sample in range(item.sample_count):  # every sample
                generator = tools.make_random_generator(seed_multiplier * sample + item.question_index)  # same seed under every configuration
                started = time.perf_counter()  # generation clock
                answer = tools.answer_service.answer_from_evidence(answer_request, evidence, configuration.layout, configuration.context_token_budget, generator)
                clock.record(time.perf_counter() - started)  # feed the projection
                records.append(build_answer_record(item, configuration, sample, answer, tools))  # keep the record
        report_progress("%s %d/%d" % (item.set_name, position, len(items)))  # one progress line per question
    return records  # in run order
