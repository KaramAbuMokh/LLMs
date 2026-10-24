# Golden sets, runtime and settings built from the shared fixture, so the stage tests run end to end without the service or any model
import dataclasses  # settings copies
from pathlib import Path
from retrieval_contracts.rare_shingle_location import ShingleHit  # gold hits
from retrieval_contracts.shared_fixture.fixture_files import FIXTURE_BUNDLE_FOLDER, load_fixture_golden_questions  # fixture
from retrieval_evaluation.evaluation_settings import CalibrationSettings, DataLocations, EvaluationSettings  # settings
from retrieval_evaluation.evaluation_stages.evaluation_runtime import EvaluationRuntime  # runtime record
from retrieval_evaluation.golden_sets.golden_question_records import (  # question records
    ABSENT_KIND, FactDevelopmentQuestion, FactQuestion, GoldenSets, HeldOutQuestion, NoAnswerQuestion, RegressionQuestion,
)
from retrieval_evaluation.metrics.fact_rule_loading import FactRule  # question plus word rules
from retrieval_evaluation.tests.support.fixture_retrieval_fakes import FakeAnswerRequest, FakeAnswerService, FakeSearchSettings, FixtureRetriever  # fakes

LAYOUT_NAMES = ("question_only", "passages_before_template", "pair_examples_before_question", "passages_and_pair_examples")  # the four layouts


def build_fixture_held_out_questions(split: str) -> tuple[HeldOutQuestion, ...]:  # fg-1..fg-4 with their gold sentences as gold hits
    return tuple(HeldOutQuestion("g1-%s-%s" % (split, golden_question.question_id), golden_question.question, golden_question.gold_answer,
                                 tuple(ShingleHit(span.part_number, span.character_start, span.character_end) for span in golden_question.gold_spans),
                                 golden_question.gold_spans[0].character_start, split) for golden_question in load_fixture_golden_questions() if golden_question.answerable)


def build_fixture_golden_sets() -> GoldenSets:  # every question set from the six fixture questions
    questions = load_fixture_golden_questions()  # fg-1..fg-6
    rule_of = lambda golden_question: FactRule(golden_question.question, ((golden_question.accepted_terms[0],),), ())  # accepted word rule
    answerable = [golden_question for golden_question in questions if golden_question.answerable]  # fg-1..fg-4
    return GoldenSets(
        build_fixture_held_out_questions("development"), build_fixture_held_out_questions("test"),
        tuple(FactQuestion("g2-%02d" % n, rule_of(golden_question), n <= 2, n == 3) for n, golden_question in enumerate(answerable[:3], start=1)),  # 3 fact questions
        tuple(FactDevelopmentQuestion("fd-%02d" % n, rule_of(golden_question), golden_question.gold_spans[0].part_number, golden_question.gold_spans[0].character_start) for n, golden_question in enumerate(answerable, start=1)),
        tuple(NoAnswerQuestion("g3-%02d" % n, golden_question.question, ABSENT_KIND) for n, golden_question in enumerate([golden_question for golden_question in questions if not golden_question.answerable], start=1)),  # fg-5, fg-6
        (RegressionQuestion("g4-01", rule_of(answerable[0])),), {"held_out_development": 4, "held_out_test": 4},
    )


def build_fake_runtime(answer_service: FakeAnswerService | None = None) -> EvaluationRuntime:  # the runtime record around the fakes
    terms = {golden_question.question: golden_question.accepted_terms[0] for golden_question in load_fixture_golden_questions() if golden_question.answerable}  # question -> accepted word
    service = answer_service or FakeAnswerService(lambda question, layout: terms.get(question, "unknown") if layout != "question_only" else "I do not know")  # grounded
    return EvaluationRuntime(FakeSearchSettings(), lambda bundle_name, settings: FixtureRetriever(settings), {"shipped": FIXTURE_BUNDLE_FOLDER}, service,
                             FakeAnswerRequest, lambda seed: seed, {name: name for name in LAYOUT_NAMES}, lambda message: None)


def build_fixture_settings(results_folder: Path) -> EvaluationSettings:  # shipped settings with a temporary results folder and a small pair support
    return dataclasses.replace(EvaluationSettings(), locations=dataclasses.replace(DataLocations(), results_folder=results_folder),
                               calibration=dataclasses.replace(CalibrationSettings(), minimum_selected_pairs=1))
