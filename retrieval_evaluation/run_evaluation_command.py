# The command: python -m retrieval_evaluation.run_evaluation_command {golden|retrieval|calibrate|answers|report|live|all}
# Run from the project folder with PYTHONPATH=.:gpt2_service_huggingface_space. This is the ONLY module that imports the Space service code
# (and torch); every stage below it receives the service objects as arguments (DESIGN.md §14.3). `all` = golden, retrieval, calibrate, answers, report.
import argparse  # command line
import dataclasses  # settings copies
import sys  # progress output
from typing import Sequence
from retrieval_evaluation.evaluation_settings import EvaluationSettings  # every path and number
from retrieval_evaluation.evaluation_stages.answers_stage import ANSWERS_FILE_NAME, run_answers_stage  # stage 4
from retrieval_evaluation.evaluation_stages.calibration_stage import CALIBRATION_FILE_NAME, run_calibration_stage  # stage 3
from retrieval_evaluation.evaluation_stages.evaluation_runtime import EvaluationRuntime  # service objects
from retrieval_evaluation.evaluation_stages.golden_sets_stage import load_golden_sets, load_rule_book, run_golden_stage  # stage 1
from retrieval_evaluation.evaluation_stages.report_stage import run_report_stage  # stage 5
from retrieval_evaluation.evaluation_stages.retrieval_stage import ABLATION_FILE_NAME, run_retrieval_stage  # stage 2
from retrieval_evaluation.experiments.evaluation_report_writing import read_json_results, write_json_results  # result files
from retrieval_evaluation.live_checks.live_contract_check import check_live_contract  # stage: live
from retrieval_evaluation.live_checks.live_latency_probe import probe_live_latency  # stage: live

OFFLINE_STAGES = ("golden", "retrieval", "calibrate", "answers", "report")  # what `all` runs, in order
SPACE_URL = "https://karamam-gpt2-demos.hf.space"  # the public Space (live stage only)
SMOKE_QUESTIONS = ("Who is the mother of dragons?", "What are the words of House Stark?", "Who is Tyrion Lannister?")  # latency probe questions


def build_evaluation_runtime(settings: EvaluationSettings, with_answers: bool) -> EvaluationRuntime:  # wires the service objects (heavy imports happen here)
    import torch  # seeded generators
    from gpt2_demo_service.grounded_answers.answer_settings import PromptLayout  # layout names
    from gpt2_demo_service.grounded_answers.book_run_index import build_book_run_index  # the copy guard's memory of the books
    from gpt2_demo_service.grounded_answers.grounded_answer_service import AnswerRequest, GroundedAnswerService  # the one answer path
    from gpt2_demo_service.index_loading.bundle_loading import load_index_bundle  # validated bundle loading
    from gpt2_demo_service.index_search.evidence_retriever import EvidenceRetriever  # search
    from gpt2_demo_service.language_models.got_text_generator import SamplingSettings, load_got_text_generator  # the model
    from gpt2_demo_service.service_settings import load_service_settings  # shipped defaults
    from retrieval_contracts.bundle_manifest import PASSAGE_RECORDS_FILE_NAME  # passage file name
    from retrieval_contracts.bundle_records import PassageRecord, read_json_lines  # stored passages
    from retrieval_encoders.pinned_pair_scorer import PinnedModelPairScorer  # reranker model
    from retrieval_encoders.pinned_text_encoder import PinnedModelTextEncoder  # dense encoder
    service = load_service_settings({})  # exactly the shipped defaults
    folders = {name: folder for name, folder in settings.locations.bundle_folders if folder.is_dir()}  # bundles that exist
    if "shipped" not in folders:  # nothing to evaluate without the real bundle
        raise FileNotFoundError("no shipped bundle at %s" % dict(settings.locations.bundle_folders)["shipped"])  # build it first
    encoder = PinnedModelTextEncoder(settings.locations.model_files_folder, thread_count=service.index_source.encoder_thread_count)  # one encoder for all
    loaded_bundles, scorers = {}, []  # caches: bundles by name, the reranker once

    def build_retriever(bundle_name, search_settings):  # a retriever on a cached bundle
        if bundle_name not in loaded_bundles:  # first use
            loaded_bundles[bundle_name] = load_index_bundle(folders[bundle_name], encoder.fingerprint)  # validates checksums too
        if search_settings.reranker_enabled and not scorers:  # the reranker model loads on first need
            scorers.append(PinnedModelPairScorer(settings.locations.model_files_folder, service.index_source.scorer_input_token_limit))  # input cap 256
        return EvidenceRetriever(loaded_bundles[bundle_name], encoder, search_settings, scorers[0] if scorers and search_settings.reranker_enabled else None)

    run = settings.answers  # generation numbers
    sampling = SamplingSettings(run.temperature, run.top_p, run.top_k, run.repetition_penalty)  # as the Space samples
    book_run_index = build_book_run_index(read_json_lines(folders["shipped"] / PASSAGE_RECORDS_FILE_NAME, PassageRecord)) if with_answers else None  # the guard sees every passage, as in the Space
    answer_service = GroundedAnswerService(  # no retriever: the stages call answer_from_evidence with their own evidence
        load_got_text_generator(settings.locations.local_weights_path, run.generator_time_limit_seconds), None, service.answers, book_run_index,
    ) if with_answers else None
    return EvaluationRuntime(  # the record every stage receives
        service.evidence_search, build_retriever, folders, answer_service, lambda question: AnswerRequest(question, run.max_new_tokens, sampling, (), True),
        lambda seed: torch.Generator().manual_seed(seed), {layout.value: layout for layout in PromptLayout}, lambda message: print(message, file=sys.stderr, flush=True),
    )


def read_required_result(settings: EvaluationSettings, file_name: str) -> dict:  # a result an earlier stage must have left
    stored = read_json_results(settings.locations.results_folder / file_name)  # None when missing
    if stored is None:  # stages depend on each other
        raise FileNotFoundError("%s is missing: run the earlier stage first" % file_name)  # clear instruction
    return stored  # parsed JSON


def run_live_stage(settings: EvaluationSettings, space_url: str) -> None:  # contract check and 20-call latency probe on the deployed Space
    problems = check_live_contract(space_url)  # empty = fine
    summary = probe_live_latency(space_url, SMOKE_QUESTIONS)  # one caller, 20 calls
    write_json_results(settings.locations.results_folder / "live_checks.json", {"contract_problems": problems, "latency": dataclasses.asdict(summary)})  # record
    print("contract problems: %s\nlatency: %s" % (problems or "none", summary))  # shown to the caller


def run_stages(stage_names: Sequence[str], settings: EvaluationSettings, space_url: str) -> None:  # runs the named stages in order
    for stage in stage_names:  # one at a time
        print("stage %s" % stage, file=sys.stderr, flush=True)  # where we are
        if stage == "golden":  # needs the data only
            run_golden_stage(settings)  # builds and writes
        elif stage == "retrieval":  # ablation and the retrieval decisions
            run_retrieval_stage(load_golden_sets(settings), build_evaluation_runtime(settings, False), settings)  # development, then one test run
        elif stage == "calibrate":  # floors and thresholds
            run_calibration_stage(load_golden_sets(settings), read_required_result(settings, ABLATION_FILE_NAME), build_evaluation_runtime(settings, False),
                                  load_rule_book(settings), settings)
        elif stage == "answers":  # generations
            run_answers_stage(load_golden_sets(settings), read_required_result(settings, ABLATION_FILE_NAME), read_required_result(settings, CALIBRATION_FILE_NAME),
                              build_evaluation_runtime(settings, True), load_rule_book(settings), settings)
        elif stage == "report":  # recommended settings and the report
            run_report_stage(settings)  # reads the result files
        elif stage == "live":  # only after a deploy
            run_live_stage(settings, space_url)  # the public Space


def main(arguments: Sequence[str] | None = None) -> int:  # entry point
    parser = argparse.ArgumentParser(description="Retrieval evaluation for the Game of Thrones model")  # help text
    parser.add_argument("stage", choices=OFFLINE_STAGES + ("live", "all"), help="stage to run; `all` runs golden, retrieval, calibrate, answers, report")  # stage
    parser.add_argument("--space-url", default=SPACE_URL, help="Space base URL for the live stage")  # live target
    parsed = parser.parse_args(arguments)  # parsed command line
    run_stages(OFFLINE_STAGES if parsed.stage == "all" else (parsed.stage,), EvaluationSettings(), parsed.space_url)  # go
    return 0  # success


if __name__ == "__main__":  # run as a module
    raise SystemExit(main())  # exit code
