# Helpers for the API tests: an app built from fakes (no 1.7 GB weights) and a fake spam model
from dataclasses import replace  # copies settings with changes
import pytest  # fixtures
from fastapi.testclient import TestClient  # in-process HTTP client
from gpt2_demo_service.grounded_answers.answer_settings import GroundedAnswerSettings, PromptLayout  # answer settings
from gpt2_demo_service.grounded_answers.grounded_answer_service import GroundedAnswerService  # the answer path
from gpt2_demo_service.index_search.evidence_retriever import EvidenceRetriever  # real retriever over the fixture
from gpt2_demo_service.service_settings import ServiceSettings, load_service_settings  # shipped defaults
from gpt2_demo_service.web_api.app_factory import ServiceComponents, create_app  # code under test
from gpt2_demo_service.service_startup.retrieval_startup import RetrievalStatus  # status record
from retrieval_contracts.shared_fixture.fixture_encoders import HashingTextEncoder  # fake encoder

SMALL_ANSWERS = GroundedAnswerSettings(  # numbers that fit the 1,024-token context easily
    layout=PromptLayout.PASSAGES_AND_PAIR_EXAMPLES, context_token_budget=300, safety_margin_tokens=8,
    answer_token_floor=8, answer_token_ceiling=20, book_snippet_enabled=True,
)


class FakeSpamClassifier:  # stands in for the spam model
    def label_email(self, email_text: str) -> str:  # labels by one keyword
        return "spam" if "winner" in email_text.lower() else "not spam"  # trivial rule


def build_ready_status() -> RetrievalStatus:  # status of a working fixture index
    return RetrievalStatus(True, "got-retrieval-20261024-eeb6e60a", "abcdef12", 6, 4, 30, "fixture/hashing-text-encoder", False, "passages_and_pair_examples", None)  # ready status of the fixture index


@pytest.fixture()
def test_settings(fixture_search_settings) -> ServiceSettings:  # shipped defaults with the fixture search numbers and small answers
    return replace(load_service_settings({}), evidence_search=fixture_search_settings, answers=SMALL_ANSWERS)  # shipped defaults with test numbers


@pytest.fixture()
def make_client(test_settings, spy_generator, fixture_bundle, fixture_search_settings):  # builds a started client; settings can be changed per test
    clients = []  # closed after the test

    def build(settings=None, retriever="fixture", status=None, generator=None):  # retriever None = retrieval off
        used_settings = settings or test_settings  # per-test settings
        real_retriever = EvidenceRetriever(fixture_bundle, HashingTextEncoder(), fixture_search_settings) if retriever == "fixture" else retriever  # search
        service = GroundedAnswerService(generator or spy_generator, real_retriever, used_settings.answers)  # answer path
        components = ServiceComponents(service, FakeSpamClassifier(), status or build_ready_status(), {"weights_repo_commit": "c117a387"})  # fakes
        client = TestClient(create_app(used_settings, lambda settings: components), raise_server_exceptions=False)  # no real weights
        client.__enter__()  # runs the startup hook
        clients.append(client)  # remember for cleanup
        return client  # the started client

    yield build  # the factory
    for client in clients:  # cleanup
        client.__exit__(None, None, None)  # runs the shutdown hook
