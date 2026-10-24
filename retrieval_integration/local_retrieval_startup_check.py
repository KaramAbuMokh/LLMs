# Starts the retrieval part of the service exactly as the Space does (private download at the pinned commit, real encoders, validation, warm-up) and asks it a few questions
# Run from the project folder: PYTHONPATH=gpt2_service_huggingface_space:. python retrieval_integration/local_retrieval_startup_check.py
import dataclasses  # settings copy with the local model folder
import json  # printed summary
import os  # the token file location from the environment
import time  # start-up and search timing
from pathlib import Path  # token and model folders
import psutil  # memory of this process
from gpt2_demo_service.service_settings import load_service_settings, read_hub_token  # shipped defaults
from gpt2_demo_service.web_api.app_factory import load_retrieval  # the real start-up path
from retrieval_evaluation.evaluation_settings import BUILD_OUTPUT_FOLDER  # where the encoder models were downloaded

TOKEN_FILE = Path(os.environ.get("HF_TOKEN_FILE", Path.home() / ".cache" / "huggingface" / "token"))  # a file holding one Hugging Face token: set HF_TOKEN_FILE, or run `huggingface-cli login`
QUESTIONS = ("Who sits on the Iron Throne?", "What are the words of House Stark?", "How many moons does Jupiter have?")  # on-topic, on-topic, off-topic


def check_retrieval_startup() -> dict:  # the status of a real start and the timing of three searches
    settings = load_service_settings({})  # code defaults
    local_source = dataclasses.replace(settings.index_source, model_files_folder=BUILD_OUTPUT_FOLDER / "model_files")  # encoder files are local on the Mac
    local_settings = dataclasses.replace(settings, index_source=local_source)  # same settings, local model folder
    started = time.perf_counter()  # start-up clock
    loaded = load_retrieval(local_settings, read_hub_token({"HF_TOKEN": TOKEN_FILE.read_text().strip()}))  # download, validate, warm up
    startup_seconds = time.perf_counter() - started  # how long the start took
    summary = {"status": dataclasses.asdict(loaded.status), "startup_seconds": round(startup_seconds, 1)}  # what we report
    if loaded.retriever is not None:  # only when retrieval is ready
        summary["searches"] = []  # one entry per question
        for question in QUESTIONS:  # each question once
            evidence = loaded.retriever.retrieve(question)  # the real search
            summary["searches"].append({"question": question, "route": str(evidence.route), "best_score": round(evidence.best_evidence_score, 3),  # no book text printed
                                        "passage_ids": [scored.record.passage_id for scored in evidence.passages], "stage_milliseconds": {name: round(value, 1) for name, value in evidence.stage_milliseconds.items()}})  # ids and timing
    summary["process_memory_mb"] = round(psutil.Process().memory_info().rss / 1e6)  # memory of the retrieval part alone
    return summary  # for printing


if __name__ == "__main__":  # run as a script
    print(json.dumps(check_retrieval_startup(), indent=1))  # readable output
