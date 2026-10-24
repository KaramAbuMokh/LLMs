# Starts retrieval at service start: loads encoders and the bundle, builds the retriever and the copy guard's index of book runs, runs one warm-up search; any failure turns retrieval OFF with a reason
import logging  # one ERROR line when retrieval cannot start
from dataclasses import dataclass  # frozen records
from pathlib import Path  # bundle and encoder folders
from gpt2_demo_service.grounded_answers.book_run_index import BookRunIndex, build_book_run_index  # every 26-word run of the books
from gpt2_demo_service.index_loading.bundle_loading import BundleValidationError, LoadedIndexBundle, load_index_bundle  # validated bundle
from gpt2_demo_service.index_search.evidence_retriever import EvidenceRetriever  # the search
from gpt2_demo_service.index_search.evidence_search_settings import EvidenceSearchSettings  # search numbers
from retrieval_contracts.encoder_contracts import PassagePairScorer, TextEncoder  # encoder interfaces

logger = logging.getLogger(__name__)  # logger for this module
WARM_UP_QUESTION = "Who is Jon Snow?"  # one search at startup so the first visitor does not pay for cold caches
INDEX_REVISION_SHORT_LENGTH = 8  # hex characters of the index commit shown in /health


@dataclass(frozen=True)  # immutable record
class RetrievalStatus:  # what /health says about retrieval
    ready: bool  # True when searches run
    index_version: str | None  # version string of the loaded bundle
    index_revision: str | None  # first 8 hex characters of the bundle's commit
    passages: int | None  # passages in the index
    pairs: int | None  # question-answer pairs in the index
    book_runs: int | None  # distinct 26-word runs the copy guard compares answers with (None when the guard has no books)
    text_encoder: str | None  # model id of the dense encoder
    reranker_enabled: bool  # the cross-encoder is in use
    prompt_layout: str  # layout the prompts use
    problem: str | None  # why retrieval is off (no text, only reasons)


@dataclass(frozen=True)  # immutable record
class RetrievalLoadResult:  # the retriever and the copy guard's index (None when off) and the status that explains them
    retriever: EvidenceRetriever | None  # ready to search, or None
    book_run_index: BookRunIndex | None  # every 26-word run of the books, or None
    status: RetrievalStatus  # for /health and the log


def describe_retrieval_off(reason: str, reranker_enabled: bool, prompt_layout: str) -> RetrievalLoadResult:  # retrieval will not run; /health says why
    logger.error("retrieval is off: %s", reason)  # one ERROR line, reasons only
    return RetrievalLoadResult(None, None, RetrievalStatus(False, None, None, None, None, None, None, reranker_enabled, prompt_layout, reason))  # no retriever, no guard index


def build_ready_status(bundle: LoadedIndexBundle, book_run_index: BookRunIndex, index_revision: str, reranker_enabled: bool, prompt_layout: str) -> RetrievalStatus:  # status of a working index
    return RetrievalStatus(  # all ten health fields
        True, bundle.manifest.index_version, index_revision[:INDEX_REVISION_SHORT_LENGTH] or None, len(bundle.passages), len(bundle.pairs),  # ready, version, short revision, passage and pair counts
        book_run_index.run_count, bundle.manifest.text_encoder.model_id, reranker_enabled, prompt_layout, None,  # guard run count, encoder, reranker switch, layout, no problem
    )


def assemble_retrieval(bundle_folder: Path, text_encoder: TextEncoder, pair_scorer: PassagePairScorer | None, search_settings: EvidenceSearchSettings,  # optional reranker and the settings that follow
                       index_revision: str, prompt_layout: str) -> RetrievalLoadResult:  # validates the bundle, builds the retriever, warms it up
    try:  # nothing here may stop the service from starting
        bundle = load_index_bundle(bundle_folder, text_encoder.fingerprint)  # checksums, versions, fingerprint, counts
        book_run_index = build_book_run_index(bundle.passages)  # the copy guard's memory of every book stretch (about 2 seconds)
        retriever = EvidenceRetriever(bundle, text_encoder, search_settings, pair_scorer)  # the search
        retriever.retrieve(WARM_UP_QUESTION)  # fails loudly now rather than on the first visitor
    except BundleValidationError as error:  # the bundle cannot be trusted
        return describe_retrieval_off(str(error), search_settings.reranker_enabled, prompt_layout)  # reasons are file names and numbers only
    except Exception as error:  # anything else: bad shapes, encoder trouble
        return describe_retrieval_off("retrieval start-up failed (%s)" % type(error).__name__, search_settings.reranker_enabled, prompt_layout)  # type name only
    return RetrievalLoadResult(retriever, book_run_index, build_ready_status(bundle, book_run_index, index_revision, search_settings.reranker_enabled, prompt_layout))  # ready


def load_encoders(model_files_folder: Path, thread_count: int, scorer_input_token_limit: int, reranker_enabled: bool) -> tuple[TextEncoder, PassagePairScorer | None]:  # loads the dense encoder and, when switched on, the reranker
    from retrieval_encoders.pinned_pair_scorer import PinnedModelPairScorer  # imported here so a missing encoder package only turns retrieval off
    from retrieval_encoders.pinned_text_encoder import PinnedModelTextEncoder  # the real dense encoder (ONNX on CPU)
    text_encoder = PinnedModelTextEncoder(model_files_folder, thread_count=thread_count)  # reads the baked files and hashes the ONNX graph
    pair_scorer = PinnedModelPairScorer(model_files_folder, scorer_input_token_limit, thread_count=thread_count) if reranker_enabled else None  # only when switched on
    return text_encoder, pair_scorer  # both encoders
