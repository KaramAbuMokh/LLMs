# Kaggle entry point of the mixed run: loads the mixed dataset and both weight files, then runs the smoke ladder (mixed_smoke)
# or the full run (mixed_full: score the original and the morning model, train from settings.start_model, score the mixed model alike)
import json  # result files
import math  # no starting best: the first evaluation is always saved
import os  # input and output overrides for local dry runs
import time  # GPU minutes
import traceback  # failure details in the report
from pathlib import Path  # file paths
import torch  # tensors and saving
from gpt_network import load_model  # the network
from mixed_evaluation import evaluate_model, validation_losses  # three-model scoring and quick checks
from mixed_examples import blocks_per_step, encode_book_blocks, mixed_stream, read_book_parts  # book blocks and the mixed stream
from mixed_ladder import check_data_and_weights, check_morning_start, check_overfit_mixed, release_memory, run_mix_sweep  # checks
from mixed_settings import build_mixed_settings  # switches
from mixed_training import MixedTrainingRun  # pairs + books per step
from model_evaluation import validation_metrics  # held-out losses
from training_examples import encode_book_windows, encode_pair, load_tokenizer, plan_epoch, read_pairs  # morning data helpers
from training_loop import save_history_plots  # curves

OUTPUT_FOLDER = Path(os.environ.get("GOT_OUTPUT_FOLDER", "/kaggle/working"))  # Kaggle keeps everything written here
STARTED = time.time()  # session start, for GPU minutes


def find_input(file_name, override_variable):  # path of an attached file under /kaggle/input (or a local override)
    if os.environ.get(override_variable):  # local dry run
        return Path(os.environ[override_variable])  # given path
    found = sorted(Path("/kaggle/input").rglob(file_name))  # every attached dataset and notebook output
    if not found:  # not attached
        raise FileNotFoundError(f"{file_name} is not attached under /kaggle/input")  # clear error in the log
    return found[0]  # the file


def write_json(file_name, content):  # one result file
    (OUTPUT_FOLDER / file_name).write_text(json.dumps(content, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def gpu_summary():  # GPU minutes so far and peak memory
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0  # gigabytes
    return {"gpu_minutes": round((time.time() - STARTED) / 60, 1), "peak_gpu_gigabytes": peak}  # both numbers


def prepare_data(settings, data_folder, tokenizer):  # pairs and book text as token examples
    train_pairs = read_pairs(data_folder, "question_answer_train.json")  # already filtered on the Mac (34,001)
    validation_pairs = read_pairs(data_folder, "question_answer_validation.json")  # the 1,233 held-out pairs, answerable flag on each
    encode = lambda pairs: [encode_pair(pair, tokenizer, settings) for pair in pairs]  # exact training prompt + answer + end token
    validation_examples = encode(validation_pairs)  # all held-out pairs
    book_parts = read_book_parts(data_folder)  # six trimmed book parts
    data = {"train_pairs": train_pairs, "validation_pairs": validation_pairs, "train_examples": encode(train_pairs),
            "validation_examples": validation_examples, "book_parts": book_parts,
            "answerable_validation_examples": [example for pair, example in zip(validation_pairs, validation_examples) if pair["answerable"]],
            "train_blocks": encode_book_blocks(book_parts, tokenizer, settings),  # 512-token blocks with unscored overlap
            "validation_windows": encode_book_windows(data_folder, "books_validation_text.txt", tokenizer, settings),  # 1024, as this morning
            "validation_text": (Path(data_folder) / "books_validation_text.txt").read_text(encoding="utf-8")}  # for the leak probe
    data["blocks_per_step"] = blocks_per_step(data["train_examples"], len(data["train_blocks"]), settings)  # about 5
    print(f"pairs {len(train_pairs)} blocks {len(data['train_blocks'])} per step {data['blocks_per_step']}", flush=True)  # log
    return data  # everything tokenised


def run_smoke(settings, data, tokenizer, paths, device):  # the ladder; the report is rewritten after every check
    checks, extra = [], {}  # results so far
    save = lambda: write_json("mixed_ladder_report.json", {"all_checks_passed": len(checks) == 4 and all(c["passed"] for c in checks),
                                                          "checks": checks, **gpu_summary(), **extra})  # report so far
    try:  # any crash is written into the report
        checks.append(check_data_and_weights(data, tokenizer, settings, paths))  # check 1
        save()  # progress
        model = load_model(settings, paths["morning"], device)  # this morning's weights (the Space's model)
        evaluation_started = time.time()  # time one full held-out pass
        checks.append(check_morning_start(validation_metrics(model, data["validation_examples"], data["validation_windows"],
                                                             settings, device), settings))  # check 2
        extra["full_evaluation_seconds"] = time.time() - evaluation_started  # for the estimate
        save()  # progress
        checks.append(check_overfit_mixed(model, data, settings, device))  # check 3 (spoils this copy)
        del model  # throw it away
        release_memory()  # room for the sweep
        save()  # progress
        checks.append(run_mix_sweep(data, tokenizer, settings, paths, device, OUTPUT_FOLDER))  # check 4
        extra["full_run_estimate"] = estimate_full_run(checks[-1], data, settings, extra["full_evaluation_seconds"])  # hours
    except Exception:  # out of memory, missing file, ...
        extra["error"] = traceback.format_exc()[-3000:]  # where it failed
        raise  # still mark the Kaggle run as failed
    finally:  # always leave the report
        save()  # final state
        print("MIXED LADDER", [(check["check"], check["passed"]) for check in checks], flush=True)  # log line


def estimate_full_run(sweep, data, settings, evaluation_seconds):  # hours for the full run from the measured step time
    chosen = sweep["details"]["chosen_question_answer_weight"] or settings.question_answer_loss_weight  # mix to time
    seconds = next(mix["seconds_per_step"] for mix in sweep["details"]["mixes"] if mix["question_answer_weight"] == chosen)  # per step
    seconds *= data["blocks_per_step"] / sweep["details"]["book_blocks_per_step"]  # full data may use a different book share
    steps = len(plan_epoch(data["train_examples"], 0, settings, settings.random_seed))  # steps per epoch
    epoch_hours = (steps * seconds + steps / settings.evaluation_every_steps * evaluation_seconds * 0.8) / 3600  # training + checks
    return {"seconds_per_step": seconds, "steps_per_epoch": steps, "hours_per_epoch": epoch_hours,
            "hours_for_max_epochs": epoch_hours * settings.max_epochs}  # the estimate


def score_weights(settings, path, data, tokenizer, device):  # loads one weight file, scores it, frees the GPU again
    model = load_model(settings, path, device)  # the weights
    scores = evaluate_model(model, data, tokenizer, settings, device)  # every number of the table
    del model  # free the GPU
    release_memory()  # room for the next model
    return scores  # the scores


def run_full(settings, data, tokenizer, paths, device):  # score the original and the morning model, train, score the mixed model
    morning = score_weights(settings, paths["morning"], data, tokenizer, device)  # this morning's model (the Space's)
    guards = [check_data_and_weights(data, tokenizer, settings, paths), check_morning_start(morning["metrics"], settings)]  # checks 1-2
    write_json("full_run_guards.json", guards)  # kept with the results
    if settings.load_current_checkpoint and not all(guard["passed"] for guard in guards):  # wrong data or weights
        raise SystemExit("guard checks failed, see full_run_guards.json")  # stop before spending GPU hours
    original = score_weights(settings, paths["original"], data, tokenizer, device)  # Karam's original
    model = load_model(settings, paths[settings.start_model], device)  # start weights for the mixed training
    total_steps = len(plan_epoch(data["train_examples"], 0, settings, settings.random_seed)) * settings.max_epochs  # ceiling
    best_path = OUTPUT_FOLDER / "model.pth"  # best weights (same format as got_model.pth)
    evaluate = lambda network: validation_losses(network, data["answerable_validation_examples"], data["validation_windows"],
                                                 settings, device)  # early stopping on the answerable held-out pairs
    run = MixedTrainingRun(model, settings, device, settings.peak_learning_rate, total_steps,
                           round(total_steps * settings.warmup_fraction), settings.question_answer_loss_weight)  # pairs + books
    history = run.run(mixed_stream(data["train_examples"], data["train_blocks"], settings, device, data["blocks_per_step"]),
                      evaluate, settings.evaluation_every_steps, lambda network: torch.save(network.state_dict(), best_path),
                      settings.early_stopping_patience, settings.max_training_hours * 3600, math.inf)  # train
    save_history_plots(history, OUTPUT_FOLDER, "mixed_run")  # curves
    write_json("training_history.json", history)  # step-by-step numbers
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))  # best weights back in
    mixed = evaluate_model(model, data, tokenizer, settings, device)  # the new model
    best = min(history["evaluations"], key=lambda record: record["answer_loss"])  # the saved checkpoint's evaluation
    write_json("comparison.json", {"original_model": original, "morning_model": morning, "mixed_model": mixed, "best_evaluation": best,
                                   "stopped_because": history["stopped_because"], "start_model": settings.start_model,
                                   "question_answer_loss_weight": settings.question_answer_loss_weight,
                                   "book_blocks_per_step": data["blocks_per_step"], "train_pairs": len(data["train_examples"]),
                                   "train_book_blocks": len(data["train_blocks"]), **gpu_summary()})  # the three-model table
    print("MIXED FULL RUN DONE", json.dumps(gpu_summary()), flush=True)  # log line


def main():  # picks the mode and runs it
    settings = build_mixed_settings()  # switches for this mode
    device = torch.device("cuda" if torch.cuda.is_available() and not settings.run_mode.startswith("dry") else "cpu")  # GPU on Kaggle
    torch.manual_seed(settings.random_seed)  # repeatable
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)  # results folder
    data_folder = find_input("book_train_parts.json", "GOT_MIXED_DATA_FILE").parent  # private mixed dataset
    real = settings.load_current_checkpoint  # dry runs use a tiny random model
    paths = {"morning": find_input("model.pth", "GOT_MORNING_CHECKPOINT_FILE") if real else None,  # morning notebook's output
             "original": find_input("got_model.pth", "GOT_ORIGINAL_CHECKPOINT_FILE") if real else None}  # Karam's original
    tokenizer = load_tokenizer(data_folder)  # GPT-2 tokenizer from the dataset
    data = prepare_data(settings, data_folder, tokenizer)  # everything tokenised
    (run_full if settings.run_mode.endswith("full") else run_smoke)(settings, data, tokenizer, paths, device)  # the run
