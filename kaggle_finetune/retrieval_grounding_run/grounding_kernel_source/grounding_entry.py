# Kaggle entry point of the retrieval-grounding run: loads the grounded sequences, the plain Q&A pairs, the book text and the live model's weights,
# trains one epoch on the 70/15/15 stream and writes model.pth (best weights), the history and a before/after table
import hashlib  # checksum of the start weights
import json  # result files
import math  # no starting best: the first evaluation is always saved
import os  # input and output overrides for local dry runs
import time  # GPU minutes
from pathlib import Path  # file paths
import torch  # tensors and saving
from gpt_network import load_model  # the network
from grounding_evaluation import grounding_validation_losses, mean_seconds_by_kind  # validation numbers
from grounding_settings import LIVE_MODEL_SHA256_PREFIX, build_grounding_settings  # switches
from grounding_stream import count_epoch_steps, grounding_stream  # the 70/15/15 stream
from mixed_examples import encode_book_blocks, read_book_parts  # 512-token book blocks with unscored overlap
from model_evaluation import answer_fixed_questions  # plain-prompt answers (stop check)
from run_settings import FIXED_QUESTIONS  # the 10 fixed questions
from training_examples import encode_book_windows, encode_pair, load_tokenizer, read_pairs  # shared data helpers
from training_loop import TrainingRun, save_history_plots  # training session and curves

OUTPUT_FOLDER = Path(os.environ.get("GOT_OUTPUT_FOLDER", "/kaggle/working"))  # Kaggle keeps everything written here
STARTED = time.time()  # session start, for GPU minutes


def find_input(file_name, override_variable):  # path of an attached file under /kaggle/input (or a local override)
    if os.environ.get(override_variable):  # local dry run
        return Path(os.environ[override_variable])  # given path
    found = sorted(Path("/kaggle/input").rglob(file_name))  # every attached dataset and notebook output
    if not found:  # not attached
        raise FileNotFoundError(f"{file_name} is not attached under /kaggle/input")  # clear error in the log
    return found[0]  # the file


def find_live_weights():  # the one model.pth whose checksum is the model live in the Space
    for candidate in sorted(Path("/kaggle/input").rglob("model.pth")):  # every attached notebook output
        digest = hashlib.sha256()  # running hash
        with open(candidate, "rb") as handle:  # binary read in 16 MB pieces
            for piece in iter(lambda: handle.read(1 << 24), b""):
                digest.update(piece)  # add the piece to the checksum
        if digest.hexdigest().startswith(LIVE_MODEL_SHA256_PREFIX):  # the right file
            return candidate  # found
    raise FileNotFoundError(f"no attached model.pth has the checksum prefix {LIVE_MODEL_SHA256_PREFIX}")  # wrong start weights: stop before spending GPU time


def write_json(file_name, content):  # one result file
    (OUTPUT_FOLDER / file_name).write_text(json.dumps(content, indent=1, ensure_ascii=False, default=str), encoding="utf-8")  # pretty JSON, text kept readable


def prepare_data(settings, plain_folder, grounded_folder, tokenizer):  # everything the stream and the validation need
    read_grounded = lambda name: json.loads((grounded_folder / name).read_text(encoding="utf-8"))  # grounded files hold ready token lists
    validation_pairs = read_pairs(plain_folder, "question_answer_validation.json")  # held-out pairs with the answerable flag
    data = {"grounded": read_grounded("grounded_train.json"), "oracle_validation": read_grounded("grounded_oracle_validation.json"),  # training and oracle validation
            "retrieved_validation": read_grounded("grounded_retrieved_validation.json"),  # development questions with the Space's evidence
            "plain_pairs": [encode_pair(pair, tokenizer, settings) for pair in read_pairs(plain_folder, "question_answer_train.json")],  # the no-evidence path
            "plain_validation": [encode_pair(pair, tokenizer, settings) for pair in validation_pairs if pair["answerable"]],  # clean held-out pairs
            "blocks": encode_book_blocks(read_book_parts(plain_folder), tokenizer, settings),  # keeps the books' voice
            "validation_windows": encode_book_windows(plain_folder, "books_validation_text.txt", tokenizer, settings)}  # held-out book passage
    if settings.run_mode.startswith("dry"):  # the local crash test uses a small slice
        for key in ("grounded", "oracle_validation", "retrieved_validation", "plain_pairs", "plain_validation", "blocks"):  # every list
            data[key] = data[key][:64]  # 64 items
    print({key: len(value) for key, value in data.items()}, flush=True)  # sizes in the log
    return data  # all lists


def run_training(settings, data, tokenizer, start_path, device):  # train from the live weights and write every result
    model = load_model(settings, start_path, device)  # start weights (random tiny model in dry mode)
    before = grounding_validation_losses(model, data, settings, device)  # the start numbers
    epoch_steps = count_epoch_steps(data["grounded"], data["plain_pairs"], data["blocks"], settings)  # steps of one epoch
    smoke = settings.run_mode.endswith("smoke") or settings.run_mode.startswith("dry")  # short runs stop early
    total_steps = settings.smoke_steps if smoke else epoch_steps * settings.max_epochs  # schedule length
    best_path = OUTPUT_FOLDER / "model.pth"  # best weights (same format as got_model.pth)
    run = TrainingRun(model, settings, device, settings.peak_learning_rate, total_steps, round(total_steps * settings.warmup_fraction))  # session
    stream = grounding_stream(data["grounded"], data["plain_pairs"], data["blocks"], settings, device)  # 70/15/15
    history = run.run(stream, lambda network: grounding_validation_losses(network, data, settings, device), settings.evaluation_every_steps,
                      lambda network: torch.save(network.state_dict(), best_path), settings.early_stopping_patience, settings.max_training_hours * 3600, math.inf)  # train
    save_history_plots(history, OUTPUT_FOLDER, "grounding_run")  # curves
    write_json("training_history.json", history)  # step-by-step numbers
    if best_path.exists():  # at least one evaluation was saved
        model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))  # best weights back in
    after = grounding_validation_losses(model, data, settings, device)  # the new numbers
    stop_answers = answer_fixed_questions(model, tokenizer, FIXED_QUESTIONS, settings, device)  # plain prompts: do answers still end by themselves
    write_json("grounding_comparison.json", {"before": before, "after": after, "stopped_because": history["stopped_because"], "total_steps": total_steps, "epoch_steps": epoch_steps,  # the whole before/after table
                                              "seconds_per_step_by_kind": mean_seconds_by_kind(history), "plain_answer_loss_rise": after["plain_answer_loss"] - before["plain_answer_loss"],
                                              "plain_loss_rise_flag": after["plain_answer_loss"] - before["plain_answer_loss"] > settings.plain_loss_rise_limit,
                                              "plain_answers_ending_by_themselves": sum(answer["stop_reason"] == "end_token" for answer in stop_answers), "plain_answers": stop_answers,
                                              "gpu_minutes": round((time.time() - STARTED) / 60, 1), "peak_gpu_gigabytes": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0})
    print("GROUNDING RUN DONE", json.dumps({"before": before, "after": after}), flush=True)  # log line


def main():  # picks the mode and runs it
    settings = build_grounding_settings()  # switches for this mode
    device = torch.device("cuda" if torch.cuda.is_available() and not settings.run_mode.startswith("dry") else "cpu")  # GPU on Kaggle
    torch.manual_seed(settings.random_seed)  # repeatable
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)  # results folder
    plain_folder = find_input("book_train_parts.json", "GOT_MIXED_DATA_FILE").parent  # the mixed dataset (plain pairs, books, tokenizer files)
    grounded_folder = find_input("grounded_train.json", "GOT_GROUNDED_DATA_FILE").parent  # the grounded dataset
    start_path = find_live_weights() if settings.load_current_checkpoint else None  # live weights, checksum checked
    tokenizer = load_tokenizer(plain_folder)  # GPT-2 tokenizer from the dataset
    run_training(settings, prepare_data(settings, plain_folder, grounded_folder, tokenizer), tokenizer, start_path, device)  # go
