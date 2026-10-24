# Shared test fixtures: the GPT-2 tokenizer, the tiny random model, a scripted generator, the shared fixture bundle and the slow-test switch
import os  # reads the opt-in switch for slow tests
import shutil  # copies the fixture bundle
from dataclasses import dataclass, field  # spy record
import pytest  # fixtures and skip markers
import tiktoken  # GPT-2 tokenizer
import torch  # tensors
from gpt2_demo_service.index_loading.bundle_loading import LoadedIndexBundle, load_index_bundle  # reads the fixture bundle
from gpt2_demo_service.index_search.evidence_search_settings import EvidenceSearchSettings  # search settings record
from gpt2_demo_service.language_models.got_text_generator import GeneratedAnswer  # generator result
from gpt2_demo_service.language_models.gpt_model import BASE_CONFIG, GPTModel  # network class
from retrieval_contracts.shared_fixture.fixture_encoders import HASHING_ENCODER_FINGERPRINT  # fingerprint of the fixture encoder
from retrieval_contracts.shared_fixture.fixture_files import FIXTURE_BUNDLE_FOLDER, load_fixture_expectations  # reference bundle and numbers

TINY_MODEL_SIZE = {"embedding_dim": 32, "layer_count": 2, "head_count": 4}  # small enough to run in milliseconds
TINY_CONTEXT_LENGTH = 192  # long enough for the packing tests, short enough for a tiny mask


def pytest_collection_modifyitems(config, items):  # skips slow tests unless the switch is set
    if os.environ.get("GOT_RUN_SLOW_TESTS") == "1":  # opted in
        return  # keep everything
    skip_slow = pytest.mark.skip(reason="set GOT_RUN_SLOW_TESTS=1 to load the real weights")  # explanation shown by pytest
    for item in items:  # every collected test
        if "slow" in item.keywords:  # marked slow
            item.add_marker(skip_slow)  # not run


@pytest.fixture(scope="session")
def gpt2_tokenizer() -> tiktoken.Encoding:  # the real GPT-2 tokenizer (cached on this machine)
    return tiktoken.get_encoding("gpt2")  # same encoding the service uses


@pytest.fixture()
def tiny_model() -> GPTModel:  # a random GPT-2 with two small layers, identical on every run
    torch.manual_seed(1234)  # same weights every time
    config = {**BASE_CONFIG, **TINY_MODEL_SIZE, "context_length": TINY_CONTEXT_LENGTH, "dropout_rate": 0.0}  # tiny shape, full vocabulary
    return GPTModel(config).eval()  # inference mode


@pytest.fixture(scope="session")
def fixture_bundle() -> LoadedIndexBundle:  # the shared fixture bundle, loaded and validated once
    return load_index_bundle(FIXTURE_BUNDLE_FOLDER, HASHING_ENCODER_FINGERPRINT)  # read-only arrays


@pytest.fixture(scope="session")
def fixture_search_settings() -> EvidenceSearchSettings:  # the fixture's candidate counts (all 6 passages), floor 0.15 and pair threshold 0.45
    return EvidenceSearchSettings(  # values from fixture_expectations.json
        passage_search_mode="hybrid", dense_candidate_count=6, lexical_candidate_count=6, fusion_constant=60, fused_candidate_count=6,  # mode and candidate counts (all 6 passages)
        reranker_enabled=False, rerank_candidate_count=6, passage_evidence_floor=0.15, rerank_evidence_floor=1.0, max_passages=3,  # reranker off, floors and passage limit
        pair_examples_enabled=True, pair_candidate_count=6, pair_example_threshold=0.45, max_pair_examples=2,  # pair search on, pair counts and threshold
    )


@pytest.fixture(scope="session")
def fixture_expectations() -> dict:  # rankings and numbers every package must reproduce
    return load_fixture_expectations()  # parsed JSON


@dataclass
class SpyGenerator:  # stands in for GotTextGenerator: records its calls and returns scripted text
    tokenizer: object  # tokenizer used to build prompts
    context_length: int = 1024  # context window reported to the packer
    text: str = "A short invented answer."  # text every call returns
    stop_reason: str = "end_token"  # stop reason every call reports
    calls: list = field(default_factory=list)  # (prompt_token_ids, max_new_tokens, stop_strings) per call

    def generate_answer(self, prompt_token_ids, max_new_tokens, sampling, stop_strings, random_generator: torch.Generator | None = None) -> GeneratedAnswer:  # records the call and returns the scripted answer
        self.calls.append((tuple(prompt_token_ids), max_new_tokens, tuple(stop_strings)))  # remember the call
        return GeneratedAnswer(self.text, 5, self.stop_reason, 2.0, 3.0)  # scripted answer


@pytest.fixture()
def spy_generator(gpt2_tokenizer) -> SpyGenerator:  # a scripted generator with the real tokenizer
    return SpyGenerator(gpt2_tokenizer)  # 1,024-token context


@pytest.fixture()
def tampered_bundle_folder(tmp_path):  # a copy of the fixture bundle with one changed byte
    destination = tmp_path / "bundle"  # where the copy lives
    shutil.copytree(FIXTURE_BUNDLE_FOLDER, destination)  # copy every file
    vectors_path = destination / "question_vectors.npy"  # a data file
    content = bytearray(vectors_path.read_bytes())  # its bytes
    content[-1] ^= 0xFF  # flip the last byte
    vectors_path.write_bytes(bytes(content))  # same size, different checksum
    return destination  # the broken bundle
