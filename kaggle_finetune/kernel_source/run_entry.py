# Kaggle entry point: loads the data and the current checkpoint, then runs the sanity ladder (smoke) or the full continued training
import gc  # frees memory between checks
import json  # result files
import os  # input and output overrides for local dry runs
import random  # slice sampling
import time  # GPU minutes
from pathlib import Path  # file paths
import torch  # tensors
import traceback  # failure details in the report
from gpt_network import load_model  # the network
from model_evaluation import answer_fixed_questions, validation_metrics  # scoring
from pair_filters import filter_training_pairs  # data review fixes
from run_settings import FIXED_QUESTIONS, build_run_settings  # switches
from sanity_ladder import (check_decoded_examples, check_generation_stops, check_overfit_tiny_batch, check_save_and_reload,
                           check_small_slice, check_starting_loss)  # the five checks plus the save test
from training_examples import encode_book_windows, encode_pair, load_tokenizer, read_pairs, stream_batches, steps_per_epoch
from training_loop import TrainingRun, estimate_full_run, save_history_plots  # training

OUTPUT_FOLDER = Path(os.environ.get("GOT_OUTPUT_FOLDER", "/kaggle/working"))  # Kaggle keeps everything written here
STARTED = time.time()  # session start, for GPU minutes


def find_input(file_name, override_variable):  # path of an input file inside /kaggle/input (or a local override)
    if os.environ.get(override_variable):  # local dry run
        return Path(os.environ[override_variable])  # given path
    return sorted(Path("/kaggle/input").rglob(file_name))[0]  # the attached private dataset


def write_json(file_name, content):  # one result file
    (OUTPUT_FOLDER / file_name).write_text(json.dumps(content, indent=1, ensure_ascii=False, default=str), encoding="utf-8")  # pretty JSON


def free_memory():  # drop unused GPU memory between stages
    gc.collect()  # Python objects
    if torch.cuda.is_available():  # on the GPU
        torch.cuda.empty_cache()  # give cached blocks back


def prepare_data(settings, data_folder, tokenizer):  # pairs filtered, encoded; book windows encoded
    train_pairs, filter_report = filter_training_pairs(read_pairs(data_folder, "question_answer_train.json"), settings)  # review fixes
    validation_pairs = read_pairs(data_folder, "question_answer_validation.json")  # held-out passage, untouched
    encode = lambda pairs: [encode_pair(pair, tokenizer, settings) for pair in pairs]  # pairs to token examples
    print("filters:", json.dumps(filter_report), flush=True)  # log
    return {"train_pairs": train_pairs, "validation_pairs": validation_pairs, "filter_report": filter_report,
            "train_examples": encode(train_pairs), "validation_examples": encode(validation_pairs),
            "train_windows": encode_book_windows(data_folder, "books_train_text.txt", tokenizer, settings),
            "validation_windows": encode_book_windows(data_folder, "books_validation_text.txt", tokenizer, settings)}


def peak_memory_gigabytes():  # highest GPU memory use so far
    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0  # 0 on the CPU


def save_ladder_report(checks, extra):  # written after every check, so a crash still shows what passed
    write_json("sanity_ladder_report.json", {"all_checks_passed": len(checks) == 5 and all(check["passed"] for check in checks),
                                             "checks": checks, "peak_gpu_gigabytes": peak_memory_gigabytes(),
                                             "gpu_minutes": (time.time() - STARTED) / 60, **extra})  # report so far


def run_smoke(settings, data, tokenizer, checkpoint_path, device):  # the five-step sanity ladder
    checks, extra = [], {"filter_report": data["filter_report"]}  # results so far
    try:  # any crash is written into the report
        run_ladder(settings, data, tokenizer, checkpoint_path, device, checks, extra)  # the checks
    except Exception:  # out of memory, shape errors, ...
        extra["error"] = traceback.format_exc()[-3000:]  # where it failed
        raise  # still mark the Kaggle run as failed
    finally:  # always leave the report
        save_ladder_report(checks, extra)  # final state
    print("SANITY LADDER", "READY" if all(check["passed"] for check in checks) and len(checks) == 5 else "NOT READY", flush=True)


def run_ladder(settings, data, tokenizer, checkpoint_path, device, checks, extra):  # checks 1 to 5 in order
    checks.append(check_decoded_examples(data["train_pairs"], data["train_examples"], data["validation_pairs"], tokenizer, settings))
    model = load_model(settings, checkpoint_path, device)  # current fine-tuned weights
    evaluation_started = time.time()  # time one full validation pass
    current_metrics = validation_metrics(model, data["validation_examples"], data["validation_windows"], settings, device)  # step 0
    evaluation_seconds = time.time() - evaluation_started  # used for the full-run estimate
    checks.append(check_starting_loss(current_metrics, settings))  # check 2
    extra["current_model_answers"] = answer_fixed_questions(model, tokenizer, FIXED_QUESTIONS, settings, device)  # before answers
    save_ladder_report(checks, extra)  # progress
    checks.append(check_overfit_tiny_batch(model, data["train_examples"], settings, device))  # check 3 (spoils the weights)
    save_ladder_report(checks, extra)  # progress
    del model  # throw the over-fitted copy away
    free_memory()  # make room
    model = load_model(settings, checkpoint_path, device)  # fresh current weights for the slice
    sampler = random.Random(settings.random_seed)  # repeatable slice
    slice_pairs = sampler.sample(data["train_examples"], settings.slice_question_answer_count)  # about 2,000 pairs
    slice_windows = sampler.sample(data["train_windows"], settings.slice_book_window_count)  # about 80 book windows
    every = max(1, len(data["validation_examples"]) // settings.slice_validation_pair_count)  # spread over the whole passage
    validation_pairs = data["validation_examples"][::every][: settings.slice_validation_pair_count]  # validation subset
    validation_windows = data["validation_windows"][: settings.slice_validation_window_count]  # book subset
    evaluate = lambda network: validation_metrics(network, validation_pairs, validation_windows, settings, device)  # quick scoring
    slice_start = evaluate(model)  # before the slice
    warmup_steps = max(1, round(settings.slice_steps * settings.warmup_fraction))  # 5% warmup
    run = TrainingRun(model, settings, device, settings.peak_learning_rate, settings.slice_steps, warmup_steps)  # real schedule
    history = run.run(stream_batches(slice_pairs, slice_windows, settings, device), evaluate, settings.slice_evaluation_every_steps)
    checks.append(check_small_slice(history, slice_start, settings))  # check 4
    write_json("smoke_slice_history.json", history)  # step-by-step numbers
    save_history_plots(history, OUTPUT_FOLDER, "smoke_slice")  # learning-rate and loss picture
    extra["full_run_estimate"] = estimate_full_run(history, data["train_examples"], len(data["train_windows"]), settings,
                                                   evaluation_seconds, steps_per_epoch)  # hours for the real run
    extra["save_and_reload"] = check_save_and_reload(model, settings, device, OUTPUT_FOLDER)  # saving works, Space-compatible
    extra["gpu"] = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"  # which card
    save_ladder_report(checks, extra)  # progress
    checks.append(check_generation_stops(answer_fixed_questions(model, tokenizer, FIXED_QUESTIONS, settings, device), settings))  # check 5


def run_full(settings, data, tokenizer, checkpoint_path, device):  # continued training from the current checkpoint + before/after
    model = load_model(settings, checkpoint_path, device)  # current fine-tuned weights
    evaluate = lambda network: validation_metrics(network, data["validation_examples"], data["validation_windows"], settings, device)
    before = evaluate(model)  # current model on the held-out passage
    guards = [check_decoded_examples(data["train_pairs"], data["train_examples"], data["validation_pairs"], tokenizer, settings),
              check_starting_loss(before, settings)]  # quick checks 1 and 2 again
    write_json("full_run_guards.json", guards)  # kept with the results
    if settings.load_current_checkpoint and not all(guard["passed"] for guard in guards):  # format or weights do not match
        raise SystemExit("guard checks failed, see full_run_guards.json")  # stop before spending GPU hours
    before_answers = answer_fixed_questions(model, tokenizer, FIXED_QUESTIONS, settings, device)  # current model answers
    pair_batches, book_batches = steps_per_epoch(data["train_examples"], len(data["train_windows"]), settings)  # epoch size
    total_steps = (pair_batches + book_batches) * settings.max_epochs  # ceiling
    best_path = OUTPUT_FOLDER / "model.pth"  # best weights (same format as got_model.pth)
    run = TrainingRun(model, settings, device, settings.peak_learning_rate, total_steps, round(total_steps * settings.warmup_fraction))
    history = run.run(stream_batches(data["train_examples"], data["train_windows"], settings, device), evaluate,
                      settings.evaluation_every_steps, lambda network: torch.save(network.state_dict(), best_path),
                      settings.early_stopping_patience, settings.max_training_hours * 3600, before["answer_loss"])
    save_history_plots(history, OUTPUT_FOLDER, "full_run")  # curves
    write_json("training_history.json", history)  # step-by-step numbers
    if best_path.exists():  # an improvement was saved
        model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))  # best weights back in
    after = evaluate(model) if best_path.exists() else None  # new model on the same passage
    after_answers = answer_fixed_questions(model, tokenizer, FIXED_QUESTIONS, settings, device) if best_path.exists() else None
    write_json("before_after.json", {"current_model": {"metrics": before, "answers": before_answers},  # the comparison
                                     "new_model": {"metrics": after, "answers": after_answers, "weights": "model.pth" if after else None},
                                     "stopped_because": history["stopped_because"], "filter_report": data["filter_report"],
                                     "gpu_minutes": (time.time() - STARTED) / 60, "peak_gpu_gigabytes": peak_memory_gigabytes()})


def main():  # picks the mode and runs it
    settings = build_run_settings()  # switches for this mode
    device = torch.device("cuda" if torch.cuda.is_available() and not settings.run_mode.startswith("dry") else "cpu")  # GPU on Kaggle
    torch.manual_seed(settings.random_seed)  # repeatable
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)  # results folder
    data_folder = find_input("question_answer_train.json", "GOT_DATA_FILE").parent  # private training-data dataset
    checkpoint_path = find_input("got_model.pth", "GOT_CHECKPOINT_FILE") if settings.load_current_checkpoint else None  # weights
    tokenizer = load_tokenizer(data_folder)  # GPT-2 tokenizer from the dataset
    data = prepare_data(settings, data_folder, tokenizer)  # everything tokenised
    (run_full if settings.run_mode in ("full", "dry_full") else run_smoke)(settings, data, tokenizer, checkpoint_path, device)  # the run
