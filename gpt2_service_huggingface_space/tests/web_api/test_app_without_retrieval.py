# Tests of the app when retrieval cannot start: the kill switch, an unpinned revision and a broken bundle keep /generate answering with the plain prompt
from dataclasses import replace  # copies settings with changes
from gpt2_demo_service.service_startup.retrieval_startup import assemble_retrieval  # builds the retrieval pieces
from gpt2_demo_service.web_api.app_factory import load_retrieval  # start-up wiring under test
from retrieval_contracts.shared_fixture.fixture_encoders import HashingTextEncoder  # fake encoder

GOT_BODY = {"prompt": "Where are the keys of the salt tower kept?", "task_name": "got", "max_new_tokens": 50}  # old portfolio body


def test_kill_switch_and_unpinned_revision_keep_retrieval_off(test_settings):  # load_retrieval never raises
    switched_off = replace(test_settings, index_source=replace(test_settings.index_source, enabled=False))  # GOT_RETRIEVAL_ENABLED=false
    assert "switched off" in load_retrieval(switched_off, None).status.problem  # reason shown
    unpinned = load_retrieval(replace(test_settings, index_source=replace(test_settings.index_source, index_revision="")), None)  # an empty revision is refused
    assert unpinned.retriever is None and unpinned.status.problem == "retrieval start-up failed (ValueError)"  # refused before any network call


def test_broken_bundle_still_serves_generate(make_client, tampered_bundle_folder, fixture_search_settings):  # D19
    off = assemble_retrieval(tampered_bundle_folder, HashingTextEncoder(), None, fixture_search_settings, "a" * 40, "passages_and_pair_examples")  # start retrieval over the broken bundle
    client = make_client(retriever=off.retriever, status=off.status)  # app with retrieval off
    body = client.post("/generate", json=GOT_BODY).json()  # ask
    assert client.get("/health").json()["retrieval"]["ready"] is False  # health says so
    assert body["retrieval_used"] is False and body["sources"] == [] and isinstance(body["completion"], str)  # today's plain answer
