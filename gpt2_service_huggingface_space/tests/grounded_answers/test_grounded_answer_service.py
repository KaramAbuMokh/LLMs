# Tests of the answer path: both entry points agree, routes follow what the prompt holds, retrieval trouble never breaks an answer, the guard and time limit act
from dataclasses import replace  # copies settings with changes
import pytest  # expected errors
import torch  # seeded generators
from gpt2_demo_service.grounded_answers.answer_settings import GroundedAnswerSettings, PromptLayout  # settings and layouts
from gpt2_demo_service.grounded_answers.book_run_index import build_book_run_index  # the copy guard's memory of the books
from gpt2_demo_service.grounded_answers.grounded_answer_service import AnswerRequest, GroundedAnswerService, QuestionTooLongError  # code under test
from gpt2_demo_service.grounded_answers.prompt_layouts import render_question_block  # the trained question block
from gpt2_demo_service.index_search.evidence_records import EvidenceRoute  # route names
from gpt2_demo_service.index_search.evidence_retriever import EvidenceRetriever  # real retriever over the fixture
from gpt2_demo_service.language_models.got_text_generator import GotTextGenerator, SamplingSettings  # real generator around the tiny model, sampling record
from retrieval_contracts.shared_fixture.fixture_encoders import HashingTextEncoder  # fake encoder

SAMPLING = SamplingSettings(temperature=0.7, top_p=0.9, top_k=40, repetition_penalty=1.1)  # the API defaults
SETTINGS = GroundedAnswerSettings(  # small numbers that fit the tiny model's 192-token context
    layout=PromptLayout.PASSAGES_AND_PAIR_EXAMPLES, context_token_budget=100, safety_margin_tokens=8,  # layout and budgets that fit the tiny context
    answer_token_floor=8, answer_token_ceiling=20, book_snippet_enabled=True,  # answer reserve limits and snippet switch
)


def make_request(question: str, max_new_tokens: int = 10, use_retrieval: bool = True) -> AnswerRequest:  # a request with the API defaults
    return AnswerRequest(question, max_new_tokens, SAMPLING, (), use_retrieval)  # no extra stop texts


def make_retriever(fixture_bundle, fixture_search_settings) -> EvidenceRetriever:  # real retriever over the fixture bundle
    return EvidenceRetriever(fixture_bundle, HashingTextEncoder(), fixture_search_settings)  # no reranker


def test_both_entry_points_agree_for_the_same_evidence_and_seed(tiny_model, gpt2_tokenizer, fixture_bundle, fixture_search_settings):  # one answer path
    generator = GotTextGenerator(tiny_model, gpt2_tokenizer, 600.0)  # tiny model, never stops on the clock
    retriever = make_retriever(fixture_bundle, fixture_search_settings)  # real search
    service = GroundedAnswerService(generator, retriever, SETTINGS)  # service under test
    request = make_request("Where are the keys of the salt tower kept?")  # fg-3
    via_question = service.answer_question(request, torch.Generator().manual_seed(21))  # retrieve inside
    evidence = retriever.retrieve(request.question)  # retrieve outside
    via_evidence = service.answer_from_evidence(request, evidence, SETTINGS.layout, SETTINGS.context_token_budget, torch.Generator().manual_seed(21))  # same seed
    assert (via_question.completion, via_question.sources, via_question.route) == (via_evidence.completion, via_evidence.sources, via_evidence.route)  # identical
    assert via_question.prompt_token_count == via_evidence.prompt_token_count and via_question.passage_ids == via_evidence.passage_ids  # same prompt
    assert via_question.retrieval_used and via_question.route == EvidenceRoute.PASSAGES_AND_PAIRS  # evidence reached the prompt


def test_retrieval_switched_off_gives_todays_prompt(spy_generator, fixture_bundle, fixture_search_settings):  # use_retrieval false
    service = GroundedAnswerService(spy_generator, make_retriever(fixture_bundle, fixture_search_settings), SETTINGS)  # service with a retriever
    answer = service.answer_question(make_request("Where are the keys of the salt tower kept?", use_retrieval=False))  # opted out
    assert answer.route == EvidenceRoute.RETRIEVAL_OFF and not answer.retrieval_used and answer.sources == ()  # plain answer
    assert spy_generator.calls[0][0] == tuple(spy_generator.tokenizer.encode(  # exactly the question block
        "Answer the following question with an appropriate response. Write a response that appropriately completes the qusetion."
        "\n\n### question:\nWhere are the keys of the salt tower kept?\n\n### Response:\n"))


def test_service_without_a_retriever_is_retrieval_off(spy_generator):  # startup could not load the bundle
    answer = GroundedAnswerService(spy_generator, None, SETTINGS).answer_question(make_request("Anything?"))  # no retriever
    assert answer.route == EvidenceRoute.RETRIEVAL_OFF and answer.completion == spy_generator.text  # plain answer


def test_unanswerable_question_gets_no_evidence(spy_generator, fixture_bundle, fixture_search_settings):  # fg-5 is below the floor
    service = GroundedAnswerService(spy_generator, make_retriever(fixture_bundle, fixture_search_settings), SETTINGS)  # service
    answer = service.answer_question(make_request("How many moons does Jupiter have?"))  # off-topic
    assert answer.route == EvidenceRoute.NO_EVIDENCE and not answer.retrieval_used and answer.sources == ()  # plain prompt


def test_a_failing_search_falls_back_to_the_plain_prompt(spy_generator):  # D19: retrieval never breaks /generate
    class BrokenRetriever:  # raises on every search
        def retrieve(self, question):  # same name as the real one
            raise RuntimeError("index exploded")  # simulated failure
    answer = GroundedAnswerService(spy_generator, BrokenRetriever(), SETTINGS).answer_question(make_request("Anything?"))  # search fails
    assert answer.route == EvidenceRoute.RETRIEVAL_FAILED and answer.completion == spy_generator.text and not answer.retrieval_used  # still answered


def test_route_follows_the_packed_prompt_not_the_search(spy_generator, fixture_bundle, fixture_search_settings):  # a layout may ignore one item kind
    retriever = make_retriever(fixture_bundle, fixture_search_settings)  # real search
    service = GroundedAnswerService(spy_generator, retriever, SETTINGS)  # service
    evidence = retriever.retrieve("Where are the keys of the salt tower kept?")  # passages and one pair
    assert evidence.route == EvidenceRoute.PASSAGES_AND_PAIRS  # what the search found
    answer = service.answer_from_evidence(make_request("Where are the keys of the salt tower kept?"), evidence, PromptLayout.PASSAGES_BEFORE_TEMPLATE, 100)  # passages only
    assert answer.route == EvidenceRoute.PASSAGES_ONLY and answer.pair_ids == () and all(source.source_type == "book_passage" for source in answer.sources)  # no pair source
    pairs_only = service.answer_from_evidence(make_request("Where are the keys of the salt tower kept?"), evidence, PromptLayout.PAIR_EXAMPLES_BEFORE_QUESTION, 100)  # examples only
    assert pairs_only.route == EvidenceRoute.PAIRS_ONLY and pairs_only.passage_ids == ()  # no passage source


def test_sources_have_one_book_snippet_and_the_pair_question(spy_generator, fixture_bundle, fixture_search_settings):  # citations of a full answer
    service = GroundedAnswerService(spy_generator, make_retriever(fixture_bundle, fixture_search_settings), SETTINGS)  # service
    answer = service.answer_question(make_request("Where are the keys of the salt tower kept?"))  # fg-3
    snippets = [source.snippet for source in answer.sources if source.source_type == "book_passage" and source.snippet]  # book text shown
    assert len(snippets) == 1 and answer.sources[0].snippet == snippets[0]  # exactly one, on the first passage
    assert answer.sources[-1].snippet == "Where does Corin keep the tower keys?"  # the pair source shows its question
    assert answer.sources[0].chapter == "Corin (chapter 1)" and answer.sources[0].book == "The Salt Tower"  # book and chapter


def test_answer_reserve_is_clamped_between_floor_and_ceiling(spy_generator):  # tokens requested by the generator
    service = GroundedAnswerService(spy_generator, None, SETTINGS)  # service
    for requested, expected in ((1, 8), (10, 10), (500, 20)):  # below the floor, inside, above the ceiling
        service.answer_question(make_request("Anything?", max_new_tokens=requested))  # ask
        assert spy_generator.calls[-1][1] == expected  # reserve passed to the generator


def test_copy_guard_cuts_a_repeated_passage_and_time_limit_still_cites(spy_generator, fixture_bundle, fixture_search_settings):  # guard and clock
    passage_text = fixture_bundle.passages[5].text  # the 33-word invented Ysolde passage
    spy_generator.text, spy_generator.stop_reason = passage_text, "time_limit"  # the model copies it and the clock runs out
    service = GroundedAnswerService(spy_generator, make_retriever(fixture_bundle, fixture_search_settings), SETTINGS, build_book_run_index(fixture_bundle.passages))  # service
    answer = service.answer_question(make_request("Who rules the Glass Coast from Lumen? Queen Ysolde"))  # a question that finds it
    assert "bp-1-002-000" in answer.passage_ids  # the copied passage was packed
    assert answer.verbatim_guard_cut and len(answer.completion.split()) <= 25  # cut to the limit
    assert answer.stop_reason == "time_limit" and answer.sources  # the text so far still gets its sources


def test_memorised_book_text_is_cut_on_every_route(spy_generator, fixture_bundle, fixture_search_settings):  # the guard knows ALL passages, not only the packed ones
    spy_generator.text = fixture_bundle.passages[5].text  # the model recites a passage it was never shown
    index = build_book_run_index(fixture_bundle.passages)  # every passage of the fixture
    service = GroundedAnswerService(spy_generator, make_retriever(fixture_bundle, fixture_search_settings), SETTINGS, index)  # service with a retriever
    plain = service.answer_question(make_request("Where are the keys of the salt tower kept?", use_retrieval=False))  # opted out of retrieval
    nothing_found = service.answer_question(make_request("How many moons does Jupiter have?"))  # no evidence passes the gate
    assert plain.route == EvidenceRoute.RETRIEVAL_OFF and nothing_found.route == EvidenceRoute.NO_EVIDENCE  # neither route packed a passage
    assert plain.verbatim_guard_cut and nothing_found.verbatim_guard_cut and len(plain.completion.split()) <= 25 and len(nothing_found.completion.split()) <= 25  # cut anyway
    unguarded = GroundedAnswerService(spy_generator, None, SETTINGS).answer_question(make_request("Anything?"))  # no index loaded
    assert not unguarded.verbatim_guard_cut and unguarded.completion == spy_generator.text  # nothing to compare with, so nothing is cut


def test_a_question_that_cannot_fit_is_refused_before_the_search(spy_generator):  # the 192-token tiny context is not used: the spy reports 1,024
    class CountingRetriever:  # fails the test if it is ever asked
        def retrieve(self, question):  # same name as the real one
            raise AssertionError("the search must not run for a question that cannot fit")  # proves "before retrieval"
    service = GroundedAnswerService(spy_generator, CountingRetriever(), SETTINGS)  # service
    with pytest.raises(QuestionTooLongError) as refusal:  # the emoji question of the live probe: 3 tokens for each character
        service.answer_question(make_request("\U0001F409" * 500))  # 1,500 tokens
    assert 800 < refusal.value.max_question_tokens < 1024 - 8 - 8  # what is left for the question text itself
    assert spy_generator.calls == []  # the model was never called


def test_a_question_at_the_limit_still_gets_an_answer_with_a_shrunk_reserve(spy_generator):  # the reserve never pushes prompt plus answer past the context
    template_tokens = len(spy_generator.tokenizer.encode(render_question_block(""), disallowed_special=()))  # the fixed header and markers
    fitting_counts = [count for count in range(1, 400) if template_tokens + len(spy_generator.tokenizer.encode("\U0001F409" * count)) + 8 + 8 <= 1024]  # emoji counts (three tokens each) that pass the floor check
    question = "\U0001F409" * fitting_counts[-1]  # the largest question that is still accepted
    service = GroundedAnswerService(spy_generator, None, replace(SETTINGS, answer_token_ceiling=200))  # a ceiling above the room left
    service.answer_question(make_request(question, max_new_tokens=200))  # asks for more than fits
    prompt_tokens, reserve = len(spy_generator.calls[-1][0]), spy_generator.calls[-1][1]  # what the model was given
    assert 8 <= reserve < 200 and prompt_tokens + reserve + SETTINGS.safety_margin_tokens <= 1024  # shrunk, never below the floor, always within the context


def test_stage_milliseconds_cover_search_packing_and_generation(spy_generator, fixture_bundle, fixture_search_settings):  # timings for Server-Timing and the log
    service = GroundedAnswerService(spy_generator, make_retriever(fixture_bundle, fixture_search_settings), SETTINGS)  # service
    stages = service.answer_question(make_request("Where are the keys of the salt tower kept?")).stage_milliseconds  # timings
    assert {"encode", "dense_search", "lexical_search", "fusion", "gate", "packing", "prefill", "decode"} <= set(stages)  # all stages present
