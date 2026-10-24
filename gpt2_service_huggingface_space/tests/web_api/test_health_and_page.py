# Tests of GET /health and GET /: old keys kept, retrieval status added, the try-it page is served
from gpt2_demo_service.service_startup.retrieval_startup import RetrievalStatus  # status record

OLD_HEALTH_KEYS = {"ok", "models", "weights_repo_commit", "memory_mb", "peak_memory_mb"}  # what the portfolio and the client example read
RETRIEVAL_KEYS = {"ready", "index_version", "index_revision", "passages", "pairs", "book_runs", "text_encoder", "reranker_enabled", "prompt_layout", "problem"}  # new block


def test_health_keeps_old_keys_and_adds_retrieval(make_client):  # the client example checks health.models.includes("got")
    body = make_client().get("/health").json()  # ask
    assert OLD_HEALTH_KEYS <= set(body) and body["ok"] is True and body["models"] == ["got", "spam"]  # old keys intact
    assert set(body["retrieval"]) == RETRIEVAL_KEYS and body["retrieval"]["ready"] is True and len(body["retrieval"]["index_revision"]) == 8  # new block


def test_health_says_why_retrieval_is_off(make_client):  # the problem text is a reason, nothing else
    off = RetrievalStatus(False, None, None, None, None, None, None, False, "passages_and_pair_examples", "question_vectors.npy does not match its checksum")  # broken bundle
    retrieval = make_client(retriever=None, status=off).get("/health").json()["retrieval"]  # ask
    assert retrieval["ready"] is False and retrieval["problem"] == "question_vectors.npy does not match its checksum"  # reason shown


def test_try_it_page_is_served_and_lists_sources(make_client):  # GET / returns the browser page
    response = make_client().get("/")  # ask
    assert response.status_code == 200 and "From the books:" in response.text and "text/html" in response.headers["content-type"]  # page with sources block
