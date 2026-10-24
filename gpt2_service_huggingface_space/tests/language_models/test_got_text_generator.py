# Tests of the answer writer: stop reasons, the time limit, the prompt-too-long refusal, repeatable seeded output
import pytest  # expected errors and monkeypatching
import torch  # tensors
from gpt2_demo_service.language_models import got_text_generator  # module whose sampler the tests replace
from gpt2_demo_service.language_models.got_text_generator import (  # code under test
    END_OF_TEXT_TOKEN_ID, GotTextGenerator, PromptTooLongError, SamplingSettings,
)

SAMPLING = SamplingSettings(temperature=0.7, top_p=0.9, top_k=40, repetition_penalty=1.1)  # the API defaults


@pytest.fixture()
def generator(tiny_model, gpt2_tokenizer) -> GotTextGenerator:  # generator around the tiny model with a generous time limit
    return GotTextGenerator(tiny_model, gpt2_tokenizer, answer_time_limit_seconds=600.0)  # never stops on the clock


def script_tokens(monkeypatch, token_ids):  # makes the sampler return the given tokens one by one
    remaining = list(token_ids)  # tokens still to hand out
    monkeypatch.setattr(got_text_generator, "pick_next_token", lambda *arguments: torch.tensor([[remaining.pop(0)]]))  # scripted choice


def test_end_token_stops_and_is_not_kept(generator, gpt2_tokenizer, monkeypatch):  # the model says it is done
    script_tokens(monkeypatch, [*gpt2_tokenizer.encode(" Winter is coming"), END_OF_TEXT_TOKEN_ID])  # words then the end token
    answer = generator.generate_answer(gpt2_tokenizer.encode("Question"), 20, SAMPLING, [])  # run
    assert answer.stop_reason == "end_token" and answer.text == "Winter is coming"  # trimmed words only


def test_marker_in_answer_cuts_it_off(generator, gpt2_tokenizer, monkeypatch):  # "###" starts an invented next block
    script_tokens(monkeypatch, gpt2_tokenizer.encode(" Fine answer.\n\n### question: more"))  # answer then a new block
    answer = generator.generate_answer(gpt2_tokenizer.encode("Question"), 40, SAMPLING, [])  # run
    assert answer.stop_reason == "stop_marker" and answer.text == "Fine answer."  # cut before the marker


def test_request_stop_text_ends_the_answer(generator, gpt2_tokenizer, monkeypatch):  # a caller-provided stop text
    script_tokens(monkeypatch, gpt2_tokenizer.encode(" One two STOP three four five"))  # words with a stop text inside
    answer = generator.generate_answer(gpt2_tokenizer.encode("Question"), 40, SAMPLING, ["STOP"])  # run
    assert answer.stop_reason == "stop_marker" and answer.text == "One two"  # cut before STOP


def test_token_limit_stops_after_the_reserve(generator, gpt2_tokenizer):  # a random model never ends by itself
    answer = generator.generate_answer(gpt2_tokenizer.encode("Question"), 7, SAMPLING, [], torch.Generator().manual_seed(3))  # run
    assert answer.stop_reason in ("token_limit", "end_token", "stop_marker")  # any natural stop
    assert answer.answer_token_count <= 7  # never more than the reserve


def test_zero_time_limit_stops_after_the_first_token(tiny_model, gpt2_tokenizer):  # the clock rule
    timed = GotTextGenerator(tiny_model, gpt2_tokenizer, answer_time_limit_seconds=0.0)  # already out of time
    answer = timed.generate_answer(gpt2_tokenizer.encode("Question"), 30, SAMPLING, [], torch.Generator().manual_seed(4))  # run
    assert answer.stop_reason in ("time_limit", "end_token", "stop_marker") and answer.answer_token_count <= 1  # one token at most
    assert answer.prefill_milliseconds >= 0.0 and answer.decode_milliseconds >= 0.0  # both timings reported


def test_prompt_plus_answer_over_the_context_is_refused(generator, gpt2_tokenizer):  # a packing bug must be loud
    prompt_ids = [220] * (generator.context_length - 5)  # almost fills the tiny context
    with pytest.raises(PromptTooLongError):  # ten answer tokens do not fit
        generator.generate_answer(prompt_ids, 10, SAMPLING, [])  # run


def test_same_seed_gives_same_answer(generator, gpt2_tokenizer):  # evaluation relies on repeatable generations
    prompt_ids = gpt2_tokenizer.encode("Who is the Hand of the King?")  # a prompt
    first = generator.generate_answer(prompt_ids, 12, SAMPLING, [], torch.Generator().manual_seed(11))  # first run
    second = generator.generate_answer(prompt_ids, 12, SAMPLING, [], torch.Generator().manual_seed(11))  # same seed
    assert first.text == second.text and first.answer_token_count == second.answer_token_count  # identical
