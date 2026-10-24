# Builds the web app: loads both models and the retrieval index at startup, wires the routes, CORS and public limits
import logging  # log format for the production app
import os  # the environment, read only in create_production_app
import threading  # lock so requests run one at a time on the 2 CPU cores
from contextlib import asynccontextmanager  # startup and shutdown hook
from dataclasses import dataclass  # frozen components record
from functools import partial  # binds the Hub token to the component loader
from pathlib import Path  # finds the try-it page next to this file
from typing import Callable, Mapping  # loader type
import torch  # sets the CPU thread count
from fastapi import FastAPI  # web framework
from fastapi.middleware.cors import CORSMiddleware  # lets the portfolio page call this API
from fastapi.responses import HTMLResponse  # serves the try-it page
from gpt2_demo_service.grounded_answers.grounded_answer_service import GroundedAnswerService  # the answer path
from gpt2_demo_service.index_loading.bundle_download import download_index_bundle  # private bundle fetch
from gpt2_demo_service.language_models.got_text_generator import load_got_text_generator  # the text model
from gpt2_demo_service.language_models.spam_classifier import SpamClassifier  # the spam model
from gpt2_demo_service.language_models.weights_download import download_weight_files  # private weights fetch
from gpt2_demo_service.monitoring.memory_logging import log_memory_usage  # memory numbers in the log
from gpt2_demo_service.service_settings import ServiceSettings, load_service_settings, read_hub_token  # the ONLY module that reads settings
from gpt2_demo_service.web_api import generate_route, health_route  # the two route modules
from gpt2_demo_service.web_api.request_limits import RequestGate  # line length and per-visitor rate limit
from gpt2_demo_service.service_startup.retrieval_startup import RetrievalLoadResult, RetrievalStatus, assemble_retrieval, describe_retrieval_off, load_encoders  # retrieval start

ALLOWED_ORIGINS = [  # websites whose pages may call this API from a browser
    "https://karamam.com", "https://www.karamam.com",  # the live portfolio
    "http://127.0.0.1:8803", "http://localhost:8803",  # the local preview of the portfolio build
]
VERCEL_ORIGIN_PATTERN = r"https://[a-z0-9-]+\.vercel\.app"  # any Vercel preview or production address
TRY_IT_PAGE_PATH = Path(__file__).with_name("try_it_page.html")  # browser page served at "/"


@dataclass(frozen=True)  # immutable record
class ServiceComponents:  # everything loaded at startup
    answer_service: GroundedAnswerService  # Game of Thrones answers
    spam_classifier: SpamClassifier  # spam labels
    retrieval_status: RetrievalStatus  # what /health says about retrieval
    weights_info: Mapping[str, str]  # which weights-repo commit the models came from

    @property  # read-only attribute
    def index_version(self) -> str | None:  # version of the loaded bundle, None when retrieval is off
        return self.retrieval_status.index_version  # None without a bundle


def load_retrieval(settings: ServiceSettings, hub_token: str | None) -> RetrievalLoadResult:  # downloads the bundle and starts the retriever, or explains why not
    source, layout = settings.index_source, str(settings.answers.layout)  # where from, and the layout shown in /health
    reranker_enabled = settings.evidence_search.reranker_enabled  # the cross-encoder switch
    if not source.enabled:  # the kill switch
        return describe_retrieval_off("switched off (GOT_RETRIEVAL_ENABLED)", reranker_enabled, layout)  # retrieval stays off by choice
    try:  # download and encoder start-up may fail without stopping the service
        bundle_folder = download_index_bundle(source.index_repo_id, source.index_revision, hub_token)  # pinned private snapshot
        text_encoder, pair_scorer = load_encoders(source.model_files_folder, source.encoder_thread_count, source.scorer_input_token_limit, reranker_enabled)  # baked encoder files are loaded here
    except Exception as error:  # unpinned revision, network, missing encoder files
        return describe_retrieval_off("retrieval start-up failed (%s)" % type(error).__name__, reranker_enabled, layout)  # type name only
    return assemble_retrieval(bundle_folder, text_encoder, pair_scorer, settings.evidence_search, source.index_revision, layout)  # validate and warm up


def load_service_components(settings: ServiceSettings, hub_token: str | None) -> ServiceComponents:  # loads both models, then retrieval
    torch.set_num_threads(settings.models.torch_thread_count)  # use the 2 CPU cores of cpu-basic
    got_weights_path, spam_weights_path = download_weight_files(  # fetch both weight files
        settings.models.weights_repo_id, settings.models.got_weights_file_name, settings.models.spam_weights_file_name, hub_token,  # both weight file names and the token
    )
    log_memory_usage("weights_downloaded")  # memory after download
    generator = load_got_text_generator(got_weights_path, settings.models.answer_time_limit_seconds)  # the text model
    log_memory_usage("got_model_loaded")  # memory after the first model
    spam_classifier = SpamClassifier(spam_weights_path)  # the classifier
    log_memory_usage("spam_model_loaded")  # memory with both models
    retrieval = load_retrieval(settings, hub_token)  # index and encoders (never raises)
    log_memory_usage("retrieval_loaded")  # memory with the index
    return ServiceComponents(  # all parts of the service
        GroundedAnswerService(generator, retrieval.retriever, settings.answers, retrieval.book_run_index), spam_classifier, retrieval.status,  # answer path with the copy guard index, then the spam model and the status
        {"weights_repo_commit": got_weights_path.parent.name[:8]},  # the cache folder name is the commit hash
    )


def create_app(settings: ServiceSettings, component_loader: Callable[[ServiceSettings], ServiceComponents]) -> FastAPI:  # tests inject fakes here
    @asynccontextmanager  # startup and shutdown hook
    async def load_components_on_startup(app: FastAPI):  # runs once before the first request
        app.state.components = component_loader(settings)  # load the models and the index
        yield  # the app serves requests here

    app = FastAPI(title="GPT-2 demos", version="3.0", lifespan=load_components_on_startup)  # the web app
    app.state.limits = settings.limits  # public-safety numbers for the routes
    app.state.request_defaults = settings.request_defaults  # values for request fields a caller leaves out
    app.state.request_gate = RequestGate(settings.limits)  # refuses new requests when the line is full or a visitor is too fast
    app.state.inference_lock = threading.Lock()  # one generation at a time
    app.add_middleware(  # browser access rules
        CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_origin_regex=VERCEL_ORIGIN_PATTERN,  # allowed origins and the Vercel pattern
        allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"], expose_headers=["Retry-After"],  # lets the page read how long to wait when busy
    )
    app.include_router(health_route.router)  # GET /health
    app.include_router(generate_route.router)  # POST /generate

    @app.get("/", response_class=HTMLResponse)  # route for the try-it page
    def show_try_it_page() -> str:  # the page visitors open to ask the Game of Thrones model questions
        return TRY_IT_PAGE_PATH.read_text(encoding="utf-8")  # plain HTML, no template

    return app  # ready for uvicorn


def create_production_app() -> FastAPI:  # for `uvicorn ... --factory`: the only place that reads the real environment
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")  # log to stdout
    settings = load_service_settings(os.environ)  # defaults plus overrides; a bad value stops the start and names the variable
    return create_app(settings, partial(load_service_components, hub_token=read_hub_token(os.environ)))  # models load at startup
