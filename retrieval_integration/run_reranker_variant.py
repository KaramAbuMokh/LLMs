# Re-runs the evaluation's retrieval and calibration stages with the Space slowdown factor measured live instead of the planning factor of 3,
# into its own results folder (the main results stay untouched). The live probe of 2026-10-24 showed the reranker costs about 265 ms p95 on the Space.
# Run from the project folder: PYTHONPATH=gpt2_service_huggingface_space:. python retrieval_integration/run_reranker_variant.py
import dataclasses  # settings copies
import shutil  # copies the question sets
from pathlib import Path  # result folder
from retrieval_evaluation.evaluation_settings import EvaluationSettings  # paths and numbers
from retrieval_evaluation.evaluation_stages.golden_sets_stage import GOLDEN_FILE_NAME  # question set file name
from retrieval_evaluation.run_evaluation_command import SPACE_URL, run_stages  # the stage runner

VARIANT_FOLDER = Path(__file__).resolve().parent / "results" / "reranker_variant"  # private results of this variant
MEASURED_SPACE_SLOWDOWN_FACTOR = 1.0  # live reranker retrieval p95 was 265 ms on the Space against 470 ms on the Mac: the planning factor 3 was too pessimistic, 1 is a safe bound


def run_reranker_variant() -> None:  # retrieval stage, then calibration, with the measured factor
    base = EvaluationSettings()  # shipped evaluation numbers
    VARIANT_FOLDER.mkdir(parents=True, exist_ok=True)  # create the folder
    shutil.copy(base.locations.results_folder / GOLDEN_FILE_NAME, VARIANT_FOLDER / GOLDEN_FILE_NAME)  # same question sets
    variant = dataclasses.replace(base, locations=dataclasses.replace(base.locations, results_folder=VARIANT_FOLDER),  # own results folder
                                  retrieval=dataclasses.replace(base.retrieval, space_slowdown_factor=MEASURED_SPACE_SLOWDOWN_FACTOR))  # measured factor
    run_stages(("retrieval", "calibrate"), variant, SPACE_URL)  # the two stages


if __name__ == "__main__":  # run as a script
    run_reranker_variant()  # do the work
