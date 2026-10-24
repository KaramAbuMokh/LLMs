# The runner retrieves once per question, calls answer_from_evidence once per configuration and sample, and seeds samples identically across configurations
from retrieval_evaluation.experiments.answer_run_planning import AnswerConfiguration, GenerationClock, build_answer_items  # plan records
from retrieval_evaluation.experiments.answer_runs import AnswerRunTools, run_answer_items  # under test
from retrieval_contracts.shared_fixture.fixture_files import load_fixture_golden_questions  # fixture questions
from retrieval_evaluation.tests.support.fixture_retrieval_fakes import FakeAnswerRequest, FakeAnswerService, FakeSearchSettings, FixtureRetriever  # spies

QUESTIONS = [(golden_question.question_id, golden_question.question) for golden_question in load_fixture_golden_questions()][:3]  # fg-1..fg-3 texts the fake retriever knows
CONFIGURATIONS = [AnswerConfiguration("question_only_600", "question_only", 600), AnswerConfiguration("passages_and_pair_examples_450", "passages_and_pair_examples", 450)]


def build_tools(retriever, service, copyright_calls):  # the tool bundle around spies
    return AnswerRunTools(retriever, service, FakeAnswerRequest, lambda seed: seed, lambda passage_ids, pair_ids: [],
                          lambda completion, sources: copyright_calls.append(completion) or [])  # records every A16 check


def test_retrieval_runs_once_per_question_and_answers_once_per_configuration_and_sample():  # DESIGN.md §14.3 spy test
    retriever, service, copyright_calls, clock = FixtureRetriever(FakeSearchSettings()), FakeAnswerService(lambda question, layout: "Gregor"), [], GenerationClock()
    items = build_answer_items("fact_development", QUESTIONS, 3)  # three questions, three samples
    records = run_answer_items(items, CONFIGURATIONS, build_tools(retriever, service, copyright_calls), clock, 1000)  # run
    assert retriever.retrieved_questions == [text for _, text in QUESTIONS]  # one retrieval per question, in order
    assert len(service.calls) == 3 * 3 * 2 == len(records) == len(clock.seconds) == len(copyright_calls)  # one call per question x sample x configuration
    for question_text in (text for _, text in QUESTIONS):  # per question
        calls = [call for call in service.calls if call[0] == question_text]  # its calls
        assert len({call[1] for call in calls}) == 1  # the SAME evidence object was reused for all of them


def test_seeds_depend_on_sample_and_question_position_only():  # identical across configurations
    service = FakeAnswerService()  # spy
    run_answer_items(build_answer_items("s", QUESTIONS, 2), CONFIGURATIONS, build_tools(FixtureRetriever(FakeSearchSettings()), service, []), GenerationClock(), 1000)
    seeds = {(call[0], call[2], call[3]): [] for call in service.calls}  # (question, layout, budget) -> seeds seen
    for call in service.calls:  # collect
        seeds[(call[0], call[2], call[3])].append(call[4])  # the generator argument is the seed in this test
    expected = {text: [index, 1000 + index] for index, (_, text) in enumerate(QUESTIONS)}  # sample 0 and sample 1
    assert all(seed_list == expected[key[0]] for key, seed_list in seeds.items())  # the same two seeds under every configuration


def test_records_hold_ids_scores_and_findings_but_no_passage_text():  # what is kept per answer
    records = run_answer_items(build_answer_items("s", QUESTIONS[:1], 1), CONFIGURATIONS, build_tools(FixtureRetriever(FakeSearchSettings()), FakeAnswerService(lambda question, layout_name: "Gregor rules"), []),
                               GenerationClock(), 1000)  # one question
    baseline, grounded = records  # question_only first
    assert not baseline.retrieval_used and baseline.passage_ids == () and baseline.route == "no_evidence"  # the baseline ignores evidence
    assert grounded.retrieval_used and grounded.passage_ids and grounded.configuration_name == "passages_and_pair_examples_450"  # grounded arm
    assert grounded.unsupported_names == ("Gregor",) and grounded.copyright_problems == ()  # invented name caught (nothing supports it), no A16 finding
