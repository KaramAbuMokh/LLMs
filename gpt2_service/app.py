# app.py
import os
import logging
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from model import get_service
from logging_utils import setup_logging, log_memory_usage

setup_logging()
logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_KEY")  # optional simple auth

app = FastAPI(
    title="GPT-2 Next-Word API",
    version="1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


def configure_cors(app_instance: FastAPI) -> None:
    logger.debug("configure_cors called")
    try:
        app_instance.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.debug("CORS middleware configured")
    except Exception:
        logger.exception("Failed to configure CORS middleware")
        raise


configure_cors(app)


@app.middleware("http")
async def restrict_to_generate_path(request: Request, call_next):
    logger.debug("restrict_to_generate_path called for %s", request.url.path)
    try:
        if request.url.path != "/generate":
            logger.warning("Blocked request to %s", request.url.path)
            return Response(status_code=403, content="Forbidden")
        logger.debug("Path allowed, proceeding to handler")
        response = await call_next(request)
        logger.debug("Handler processed request successfully")
        return response
    except Exception:
        logger.exception("Error in restrict_to_generate_path")
        raise HTTPException(status_code=500, detail="Internal server error")

class GenerateIn(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    task_name: str = Field(..., description="Task type: 'spam' for classification, 'got' for text generation")
    max_new_tokens: int = Field(32, ge=1, le=64)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    top_k: int = Field(50, ge=1, le=100)
    stop: list[str] = []

class GenerateOut(BaseModel):
    completion: str

@app.post("/generate", response_model=GenerateOut)
def generate(body: GenerateIn, x_api_key: str | None = None):
    logger.debug("Entered generate with task_name=%s", body.task_name)
    # tiny auth to prevent abuse
    if API_KEY and x_api_key != API_KEY:
        logger.warning("Unauthorized access attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")
    log_memory_usage("before_get_service")
    service = get_service(body.task_name)
    log_memory_usage("after_get_service")
    try:
        logger.debug("Calling service.generate")
        text = service.generate(
            prompt=body.prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            stop=body.stop or None
        )
        logger.debug("Service.generate completed")
        log_memory_usage("after_generate")
        return GenerateOut(completion=text)
    except Exception:
        logger.exception("Text generation failed")
        raise HTTPException(status_code=500, detail="Internal server error")
