# GET /health: the old keys (models, weights commit, memory) plus the retrieval status
from dataclasses import asdict  # status record to plain dictionary
from fastapi import APIRouter, Request  # routing
from gpt2_demo_service.monitoring.memory_logging import read_memory_megabytes  # memory numbers

router = APIRouter()  # included by the app factory


@router.get("/health")
def check_health(http_request: Request) -> dict:  # tells the caller the service is up, which models are loaded and what retrieval is doing
    components = http_request.app.state.components  # filled by the app's startup hook
    current_megabytes, peak_megabytes = read_memory_megabytes()  # memory numbers
    return {  # old keys first, retrieval added
        "ok": True, "models": ["got", "spam"], **components.weights_info,
        "memory_mb": current_megabytes, "peak_memory_mb": peak_megabytes,
        "retrieval": asdict(components.retrieval_status),
    }
