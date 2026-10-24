# Tests of the settings module: shipped defaults, each environment override, and bad values naming their variable
from dataclasses import replace  # copies settings with changes
import pytest  # expected errors
from gpt2_demo_service.grounded_answers.answer_settings import PromptLayout  # layout names
from gpt2_demo_service.service_settings import (  # code under test
    DEFAULT_EVIDENCE_SEARCH, DEFAULT_GROUNDED_ANSWERS, DEFAULT_PUBLIC_LIMITS, DEFAULT_REQUEST_VALUES, IndexSourceSettings, ModelSettings, ServiceSettings, load_service_settings,
    read_hub_token,
)

COMMIT = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # a well-formed commit hash


def test_empty_environment_gives_exactly_the_shipped_defaults():  # the evaluation starts from this
    assert load_service_settings({}) == ServiceSettings(IndexSourceSettings(), ModelSettings(), DEFAULT_EVIDENCE_SEARCH, DEFAULT_GROUNDED_ANSWERS, DEFAULT_PUBLIC_LIMITS, DEFAULT_REQUEST_VALUES)  # identical


def test_documented_defaults():  # numbers from the design
    settings = load_service_settings({})  # shipped
    assert settings.limits.max_requests_in_line == 5 and settings.models.answer_time_limit_seconds == 25.0 and settings.index_source.scorer_input_token_limit == 256  # D24, D8
    assert settings.evidence_search.reranker_enabled is True and settings.evidence_search.rerank_candidate_count == 10 and settings.answers.book_snippet_enabled is True  # switches (the reranker was switched on after the live cost check)
    assert settings.answers.context_token_budget == 600 and settings.models.got_weights_file_name == "got_model_retrieval_grounded.pth"  # budget and the grounding fine-tune's weights file
    assert (settings.evidence_search.passage_evidence_floor, settings.evidence_search.rerank_evidence_floor) == (0.636, 0.102)  # calibrated floors
    assert settings.evidence_search.pair_examples_enabled is False and settings.answers.layout == PromptLayout.PASSAGES_BEFORE_TEMPLATE  # pairs are off by an explicit flag, and the live layout shows none


def test_every_override_parses():  # all nine variables (the tenth, GOT_PAIR_EXAMPLES_ENABLED, has its own tests below)
    settings = load_service_settings({  # every documented variable set
        "GOT_RETRIEVAL_ENABLED": "false", "GOT_RETRIEVAL_INDEX_REVISION": COMMIT.upper(), "GOT_RETRIEVAL_RERANKER_ENABLED": "TRUE",
        "GOT_PROMPT_LAYOUT": "question_only", "GOT_BOOK_SNIPPET_ENABLED": "0", "GOT_MAX_REQUESTS_IN_LINE": "2", "TORCH_THREADS": "4",
        "WEIGHTS_REPO_ID": "owner/other-weights", "GOT_WEIGHTS_FILE_NAME": "got_model_grounded.pth",
    })
    assert (settings.index_source.enabled, settings.index_source.index_revision) == (False, COMMIT)  # kill switch and pinned commit
    assert settings.evidence_search.reranker_enabled is True and settings.answers.layout == PromptLayout.QUESTION_ONLY and settings.answers.book_snippet_enabled is False  # switches
    assert (settings.limits.max_requests_in_line, settings.models.torch_thread_count) == (2, 4)  # numbers
    assert (settings.models.weights_repo_id, settings.models.got_weights_file_name) == ("owner/other-weights", "got_model_grounded.pth")  # weights


@pytest.mark.parametrize("name,value", [
    ("GOT_RETRIEVAL_ENABLED", "maybe"), ("GOT_RETRIEVAL_RERANKER_ENABLED", "2"), ("GOT_BOOK_SNIPPET_ENABLED", "nope"),
    ("GOT_MAX_REQUESTS_IN_LINE", "0"), ("GOT_MAX_REQUESTS_IN_LINE", "three"), ("TORCH_THREADS", "-1"),
    ("GOT_PROMPT_LAYOUT", "everything"), ("GOT_RETRIEVAL_INDEX_REVISION", "main"), ("GOT_RETRIEVAL_INDEX_REVISION", COMMIT[:10]),
])
def test_bad_value_raises_naming_the_variable(name, value):  # startup stops with a clear message
    with pytest.raises(ValueError, match=name):  # the variable is named
        load_service_settings({name: value})  # bad value


def test_empty_values_keep_the_defaults():  # a blank Space variable means unset
    assert load_service_settings({"GOT_MAX_REQUESTS_IN_LINE": "", "WEIGHTS_REPO_ID": ""}) == load_service_settings({})  # unchanged


def test_hub_token_is_read_separately_and_never_stored():  # HF_TOKEN
    assert read_hub_token({"HF_TOKEN": "hf_example"}) == "hf_example" and read_hub_token({}) is None and read_hub_token({"HF_TOKEN": ""}) is None  # three cases
    assert "hf_example" not in repr(load_service_settings({"HF_TOKEN": "hf_example"}))  # settings never hold it


def test_pair_search_runs_only_when_requested_and_the_layout_shows_pairs():  # one explicit flag, no magic threshold
    pair_layout = {"GOT_PROMPT_LAYOUT": "passages_and_pair_examples"}  # a layout that shows solved examples
    assert load_service_settings({**pair_layout, "GOT_PAIR_EXAMPLES_ENABLED": "true"}).evidence_search.pair_examples_enabled is True  # both asked for
    assert load_service_settings({"GOT_PAIR_EXAMPLES_ENABLED": "true"}).evidence_search.pair_examples_enabled is False  # the live layout shows none, so the search is skipped
    with pytest.raises(ValueError, match="GOT_PAIR_EXAMPLES_ENABLED"):  # a pair layout without the pair search is a mistake, said at startup
        load_service_settings(pair_layout)  # layout alone
    with pytest.raises(ValueError, match="GOT_PAIR_EXAMPLES_ENABLED"):  # a bad value names its variable
        load_service_settings({"GOT_PAIR_EXAMPLES_ENABLED": "sometimes"})  # not a boolean


def test_server_defaults_for_fields_a_request_leaves_out():  # the portfolio sends none of them, so these decide how it sounds
    defaults = load_service_settings({}).request_defaults  # shipped values
    assert (defaults.max_new_tokens, defaults.top_p, defaults.top_k, defaults.repetition_penalty) == (120, 0.9, 40, 1.1)  # unchanged sampling numbers
    assert defaults.temperature == 0.3  # chosen by the temperature comparison of 2026-10-25 (retrieval_integration/compare_server_temperatures.py)
