# Every default value of the service in ONE place, and the only reader of the environment (through the mapping given to load_service_settings)
from dataclasses import dataclass, replace  # frozen settings records
from pathlib import Path  # baked encoder folder
from typing import Mapping  # environment mapping
from gpt2_demo_service.grounded_answers.answer_settings import GroundedAnswerSettings, PromptLayout  # answer settings and layout names
from gpt2_demo_service.grounded_answers.prompt_layouts import PAIR_LAYOUTS  # layouts that show solved pair examples
from gpt2_demo_service.index_search.evidence_search_settings import EvidenceSearchSettings  # search settings
from gpt2_demo_service.settings_from_environment import (  # typed readers that name the variable in every error
    read_boolean_variable, read_choice_variable, read_commit_variable, read_positive_integer_variable, read_text_variable,  # the five typed readers
)
from gpt2_demo_service.web_api.api_schemas import RequestDefaults  # server values for request fields left out
from gpt2_demo_service.web_api.request_limits import PublicLimitSettings  # public-safety settings

PINNED_INDEX_REVISION = "e2925f0e41b5a19a9c1e0bab09e1c338a58f2af5"  # full commit of the private index bundle (index got-retrieval-20261024-241316fb, 128-token passages); GOT_RETRIEVAL_INDEX_REVISION overrides it


@dataclass(frozen=True)  # immutable record
class IndexSourceSettings:  # where the bundle and the encoder files come from
    enabled: bool = True  # env GOT_RETRIEVAL_ENABLED: the kill switch
    index_repo_id: str = "karamAM/gpt2-got-retrieval-index"  # private dataset repo holding the bundle
    index_revision: str = PINNED_INDEX_REVISION  # env GOT_RETRIEVAL_INDEX_REVISION: full commit hash of the bundle
    model_files_folder: Path = Path("/home/user/model_files")  # public encoder files baked into the image
    encoder_thread_count: int = 2  # onnxruntime threads (cpu-basic has 2 vCPU)
    scorer_input_token_limit: int = 256  # reranker input cap in word pieces (DESIGN.md D8)


@dataclass(frozen=True)  # immutable record
class ModelSettings:  # the two GPT-2 models
    torch_thread_count: int = 2  # env TORCH_THREADS
    weights_repo_id: str = "karamAM/gpt2-got-and-spam-weights"  # env WEIGHTS_REPO_ID: private repo holding both weight files
    got_weights_file_name: str = "got_model_retrieval_grounded.pth"  # env GOT_WEIGHTS_FILE_NAME: the grounding fine-tune of 2026-10-24; rollback = "got_model.pth" (the books + Q&A mixed model)
    spam_weights_file_name: str = "spam_model.pth"  # the spam classifier
    answer_time_limit_seconds: float = 25.0  # an answer stops here so the 60-second browser client never gives up first (D24)


DEFAULT_EVIDENCE_SEARCH = EvidenceSearchSettings(  # numbers chosen by the evaluation of 2026-10-24 (hybrid search, cross-encoder rerank of the top 10, calibrated floor, no pair examples)
    passage_search_mode="hybrid", dense_candidate_count=50, lexical_candidate_count=50, fusion_constant=60,  # search mode, candidate counts and the fusion constant
    fused_candidate_count=20, reranker_enabled=True,  # env GOT_RETRIEVAL_RERANKER_ENABLED: +11.6 points of hit-at-3 for about 230 ms on the Space (measured live)
    rerank_candidate_count=10, passage_evidence_floor=0.636, rerank_evidence_floor=0.102, max_passages=3,  # rerank floor: 90 percent of development and all fact-development questions keep evidence; the dense floor is the fallback gate
    pair_examples_enabled=False,  # env GOT_PAIR_EXAMPLES_ENABLED: off, because no similarity level made a stored neighbour's answer reliable (best precision 0.47 at 0.95); the live layout shows no pairs, so the pair search is skipped
    pair_candidate_count=10, pair_example_threshold=0.95, max_pair_examples=2,  # the threshold is read only when the pair search is on
)
DEFAULT_REQUEST_VALUES = RequestDefaults(  # what a request gets for every field it leaves out (the portfolio sends only prompt, task_name and max_new_tokens); temperature 0.3 chosen on the fact-development set (14 right against 13 at 0.7, 44 samples against 41) and confirmed on the 30 facts (17 against 17, 51 samples against 47, 6 flip-prone questions against 10)
    max_new_tokens=120, temperature=0.3, top_p=0.9, top_k=40, repetition_penalty=1.1,  # answer length, randomness, nucleus size, candidate count and repeat penalty
)
DEFAULT_GROUNDED_ANSWERS = GroundedAnswerSettings(  # prompt layout, token budgets and snippet switch below
    layout=PromptLayout.PASSAGES_BEFORE_TEMPLATE,  # env GOT_PROMPT_LAYOUT: raw book passages, then the trained question template
    context_token_budget=600, safety_margin_tokens=8, answer_token_floor=120, answer_token_ceiling=200,  # retrieved-token budget, safety margin and the answer reserve limits
    book_snippet_enabled=True,  # env GOT_BOOK_SNIPPET_ENABLED
)
DEFAULT_PUBLIC_LIMITS = PublicLimitSettings(  # public-safety numbers below
    max_requests_in_line=5,  # env GOT_MAX_REQUESTS_IN_LINE: the largest cap with cap x live p95 under 60 seconds (live p95 10.1 s on 2026-10-25: 5 x 10.1 = 50 s)
    max_requests_per_visitor=20, rate_window_seconds=60, busy_retry_seconds=15, rate_retry_seconds=30,  # per-visitor allowance, its window and the two retry hints
    got_question_max_characters=500, email_max_characters=2000,  # longest question and longest email accepted
)


@dataclass(frozen=True)  # immutable record
class ServiceSettings:  # everything the service needs, assembled once at startup
    index_source: IndexSourceSettings  # bundle and encoder sources
    models: ModelSettings  # the two models
    evidence_search: EvidenceSearchSettings  # search numbers
    answers: GroundedAnswerSettings  # prompt and citation numbers
    limits: PublicLimitSettings  # public-safety numbers
    request_defaults: RequestDefaults  # values for request fields left out


def load_service_settings(environment: Mapping[str, str]) -> ServiceSettings:  # defaults plus the documented overrides; a bad value raises ValueError naming the variable
    index_source, models = IndexSourceSettings(), ModelSettings()  # shipped defaults
    answers = replace(DEFAULT_GROUNDED_ANSWERS, layout=read_choice_variable(environment, "GOT_PROMPT_LAYOUT", DEFAULT_GROUNDED_ANSWERS.layout),  # prompt layout and snippet switch with their overrides
                      book_snippet_enabled=read_boolean_variable(environment, "GOT_BOOK_SNIPPET_ENABLED", DEFAULT_GROUNDED_ANSWERS.book_snippet_enabled))  # prompt layout and snippet switch
    pairs_requested = read_boolean_variable(environment, "GOT_PAIR_EXAMPLES_ENABLED", DEFAULT_EVIDENCE_SEARCH.pair_examples_enabled)  # the pair-search switch
    if answers.layout in PAIR_LAYOUTS and not pairs_requested:  # a layout that shows pair examples cannot work without them
        raise ValueError("GOT_PROMPT_LAYOUT=%s shows pair examples, so GOT_PAIR_EXAMPLES_ENABLED must be true" % answers.layout.value)  # names both variables
    return ServiceSettings(  # the six settings groups with their overrides applied
        replace(index_source, enabled=read_boolean_variable(environment, "GOT_RETRIEVAL_ENABLED", index_source.enabled),  # index source: kill switch
                index_revision=read_commit_variable(environment, "GOT_RETRIEVAL_INDEX_REVISION", index_source.index_revision)),  # index source: pinned bundle commit
        replace(models, torch_thread_count=read_positive_integer_variable(environment, "TORCH_THREADS", models.torch_thread_count),  # models: thread count
                weights_repo_id=read_text_variable(environment, "WEIGHTS_REPO_ID", models.weights_repo_id),  # models: weights repository
                got_weights_file_name=read_text_variable(environment, "GOT_WEIGHTS_FILE_NAME", models.got_weights_file_name)),  # models: text-model weights file
        replace(DEFAULT_EVIDENCE_SEARCH, reranker_enabled=read_boolean_variable(environment, "GOT_RETRIEVAL_RERANKER_ENABLED", DEFAULT_EVIDENCE_SEARCH.reranker_enabled),  # search settings: reranker switch
                pair_examples_enabled=pairs_requested and answers.layout in PAIR_LAYOUTS),  # the search runs only for a layout that shows pairs
        answers,  # answer settings built above
        replace(DEFAULT_PUBLIC_LIMITS, max_requests_in_line=read_positive_integer_variable(environment, "GOT_MAX_REQUESTS_IN_LINE", DEFAULT_PUBLIC_LIMITS.max_requests_in_line)),  # public limits: line length
        DEFAULT_REQUEST_VALUES,  # request defaults, no overrides
    )


def read_hub_token(environment: Mapping[str, str]) -> str | None:  # HF_TOKEN: never stored in ServiceSettings, never logged
    return environment.get("HF_TOKEN") or None  # None when unset or empty
