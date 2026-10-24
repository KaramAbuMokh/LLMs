# The mixed run's smoke ladder (small samples first, Karam's standing wish); each check returns {"check", "passed", "details"}:
# 1 data and weights, 2 the morning model's start numbers, 3 a tiny mixed batch is memorised, 4 a short run of each tried mix
# keeps answers stopping and does not hurt the held-out losses (and picks the mix for the full run).
import gc  # frees memory between tried mixes
import hashlib  # checkpoint fingerprints
import math  # finite checks
import random  # slice sampling
import statistics  # means
import torch  # GPU cache
from gpt_network import load_model  # the network
from mixed_evaluation import validation_losses  # quick held-out numbers
from mixed_examples import blocks_have_no_end_token, book_token_shares, mixed_stream, single_batch_forever  # book blocks and streams
from mixed_settings import MORNING_MODEL_SHA256, ORIGINAL_MODEL_SHA256  # expected fingerprints
from mixed_training import MixedTrainingRun  # pairs + books per step
from model_evaluation import answer_fixed_questions  # the 10 fixed questions
from run_settings import FIXED_QUESTIONS  # their text
from sanity_ladder import check_decoded_examples, check_save_and_reload, learning_rate_shape_is_right  # morning checks reused
from training_examples import IGNORED_LABEL, make_targets  # label masking


def release_memory():  # drop unused GPU memory
    gc.collect()  # Python objects
    if torch.cuda.is_available():  # on the GPU
        torch.cuda.empty_cache()  # give cached blocks back


def file_sha256(path):  # SHA-256 of a weights file, read in 16 MB pieces
    digest = hashlib.sha256()  # running hash
    with open(path, "rb") as handle:  # binary read
        for piece in iter(lambda: handle.read(1 << 24), b""):  # until the end of the file
            digest.update(piece)  # add the piece
    return digest.hexdigest()  # hex text


def check_data_and_weights(data, tokenizer, settings, paths):  # check 1
    pair_check = check_decoded_examples(data["train_pairs"], data["train_examples"], data["validation_pairs"], tokenizer, settings)
    first_blocks = [block for block in data["train_blocks"] if block["part"] == 0][:3]  # three blocks in a row of the first book
    scored = [label for block in first_blocks for label in make_targets(block, True)[1] if label != IGNORED_LABEL]  # what the loss sees
    part_ids = tokenizer.encode_ordinary(data["book_parts"][0]["text"])  # the same book as tokens
    probes = [data["validation_text"][start:start + 300] for start in range(1000, len(data["validation_text"]) - 300, 25000)]
    verdicts = {"pair_format": pair_check["passed"],  # exact training prompt, one end token, answer-only labels
                "book_blocks_cover_text_once": scored == part_ids[1:1 + len(scored)],  # overlap unscored: no gap, no double count
                "no_end_token_in_book_blocks": blocks_have_no_end_token(data["train_blocks"]),  # books never teach stopping
                "held_out_passage_not_in_training_books": not any(probe in part["text"] for probe in probes for part in data["book_parts"]),
                "morning_weights_match": paths["morning"] is None or file_sha256(paths["morning"]) == MORNING_MODEL_SHA256,  # Space model
                "original_weights_match": paths["original"] is None or file_sha256(paths["original"]) == ORIGINAL_MODEL_SHA256}
    details = {**verdicts, "pair_examples": pair_check["details"]["examples"][:2], "train_pairs": len(data["train_examples"]),
               "train_book_blocks": len(data["train_blocks"]), "book_blocks_per_step": data["blocks_per_step"],
               "book_token_shares": book_token_shares(data["train_blocks"], settings),
               "validation_pairs": len(data["validation_examples"]), "answerable_validation_pairs": len(data["answerable_validation_examples"]),
               "first_book_block_text": tokenizer.decode(first_blocks[0]["tokens"][:80]), "passage_probes": len(probes)}
    return {"check": "1_data_and_weights", "passed": all(verdicts.values()), "details": details}  # pass + what was seen


def check_morning_start(metrics, settings):  # check 2: the morning model scores what it scored at 10:41 (right weights, same data)
    answer_gap = abs(metrics["answer_loss"] - settings.expected_morning_answer_loss)  # all 1,233 held-out pairs
    book_gap = abs(metrics["book_loss"] - settings.expected_morning_book_loss)  # held-out book passage
    passed = answer_gap < settings.starting_loss_tolerance and book_gap < settings.starting_loss_tolerance  # both close
    return {"check": "2_morning_model_start", "passed": passed, "details": {**metrics, "answer_gap": answer_gap, "book_gap": book_gap}}


def check_overfit_mixed(model, data, settings, device):  # check 3: one tiny mixed batch (8 pairs + 2 book blocks) is memorised
    pairs, blocks = data["train_examples"][: settings.overfit_pair_count], data["train_blocks"][: settings.overfit_block_count]
    run = MixedTrainingRun(model, settings, device, settings.overfit_learning_rate, settings.overfit_steps, 10, 0.5)  # half and half
    history = run.run(single_batch_forever(pairs, blocks, settings, device))  # same batch every step
    pair_losses, book_losses = history["pair_loss"], history["book_loss"]  # per-source curves
    passed = (all(math.isfinite(loss) for loss in pair_losses + book_losses) and pair_losses[-1] < settings.overfit_pass_loss
              and book_losses[-1] < settings.overfit_pass_loss)  # both fell close to zero
    details = {"pair_loss_start": pair_losses[0], "pair_loss_end": pair_losses[-1], "book_loss_start": book_losses[0],
               "book_loss_end": book_losses[-1], "pair_loss_every_20_steps": pair_losses[::20], "book_loss_every_20_steps": book_losses[::20]}
    return {"check": "3_overfit_tiny_mixed_batch", "passed": passed, "details": details}  # pass + curves


def judge_mix(weight, history, start, end, answers, settings):  # verdicts for one tried mix
    skipped = set(history["overflow_skipped_steps"])  # fp16 calibration steps (their norm is inf by design)
    norms = [norm for step, norm in zip(history["step"], history["gradient_norm"]) if step not in skipped]  # real updates
    tenth = max(2, len(history["pair_loss"]) // 10)  # first and last tenth of the run
    schedule_right, _ = learning_rate_shape_is_right(history["learning_rate"], settings)  # warmup then cosine
    stops = sum(answer["stop_reason"] == "end_token" for answer in answers)  # answers that ended by themselves
    verdicts = {"no_nan": history["non_finite_losses"] == 0, "learning_rate_warmup_then_cosine": schedule_right,
                "gradient_norms_sane": all(math.isfinite(norm) for norm in norms) and max(norms) < settings.max_gradient_norm_sane
                and len(skipped) <= 0.1 * len(history["step"]), "answers_still_stop": stops >= settings.stop_pass_count,
                "book_validation_loss_fell": end["book_loss"] < start["book_loss"],
                "answer_validation_loss_not_hurt": end["answer_loss"] < start["answer_loss"] + 0.03}  # the checks that must hold
    mean = statistics.mean  # short name
    return {"question_answer_weight": weight, "passed": all(verdicts.values()), **verdicts, "stops": stops,
            "answer_loss_end": end["answer_loss"], "book_loss_end": end["book_loss"],
            "pair_train_loss_first_last_tenth": [mean(history["pair_loss"][:tenth]), mean(history["pair_loss"][-tenth:])],
            "book_train_loss_first_last_tenth": [mean(history["book_loss"][:tenth]), mean(history["book_loss"][-tenth:])],
            "gradient_norm_median": statistics.median(norms), "gradient_norm_max": max(norms),
            "seconds_per_step": mean(history["step_seconds"][min(10, len(history["step_seconds"]) - 1):]), "answers": answers}


def pick_mix(results, settings):  # lowest held-out answer + book loss among passing mixes (learn both); near-ties go to more books
    passing = [result for result in results if result["passed"]]  # mixes that kept everything sane
    if not passing:  # none did
        return None  # the ladder fails
    combined = lambda result: result["answer_loss_end"] + result["book_loss_end"]  # held-out loss of a half-and-half mix (x2)
    best = min(combined(result) for result in passing)  # lowest combined held-out loss
    tied = [result for result in passing if combined(result) <= best + settings.sweep_tie_margin]  # within the tie margin
    return min(tied, key=lambda result: result["question_answer_weight"])["question_answer_weight"]  # least pair weight = most books


def run_mix_sweep(data, tokenizer, settings, paths, device, output_folder):  # check 4 (plus the save-and-reload test)
    sampler = random.Random(settings.random_seed)  # repeatable slice
    pairs = sampler.sample(data["train_examples"], min(settings.sweep_pair_count, len(data["train_examples"])))  # one pass of pairs
    blocks = sampler.sample(data["train_blocks"], min(settings.sweep_block_count, len(data["train_blocks"])))  # book blocks
    answerable = data["answerable_validation_examples"]  # clean held-out pairs
    validation_pairs = answerable[:: max(1, len(answerable) // settings.sweep_validation_pair_count)][: settings.sweep_validation_pair_count]
    windows = data["validation_windows"][: settings.sweep_validation_window_count]  # held-out book subset
    each_step = max(1, round(len(blocks) * settings.question_answer_batch_size / len(pairs)))  # books and pairs end together
    results, start, reload_test = [], None, None  # per-mix results, shared start numbers, save test
    for weight in settings.sweep_loss_weights:  # every tried mix starts from the same weights again
        model = load_model(settings, paths[settings.start_model], device)  # fresh start weights (original or morning)
        start = start or validation_losses(model, validation_pairs, windows, settings, device)  # same start for every mix
        run = MixedTrainingRun(model, settings, device, settings.peak_learning_rate, settings.sweep_steps,
                               max(1, round(settings.sweep_steps * settings.warmup_fraction)), weight)  # real schedule, compressed
        history = run.run(mixed_stream(pairs, blocks, settings, device, each_step))  # short mixed run
        end = validation_losses(model, validation_pairs, windows, settings, device)  # after it
        answers = answer_fixed_questions(model, tokenizer, FIXED_QUESTIONS, settings, device)  # do answers still stop?
        results.append(judge_mix(weight, history, start, end, answers, settings))  # verdicts
        if weight == settings.sweep_loss_weights[-1]:  # last mix: saving works and loads strictly
            reload_test = check_save_and_reload(model, settings, device, output_folder)  # Space-compatible file
        del model, run  # free the copy and its optimizer
        release_memory()  # room for the next one
    chosen = pick_mix(results, settings)  # the mix for the full run
    details = {"start_model": settings.start_model, "start": start, "mixes": results, "chosen_question_answer_weight": chosen,
               "book_blocks_per_step": each_step,
               "save_and_reload": reload_test, "slice": {"pairs": len(pairs), "blocks": len(blocks), "validation_pairs": len(validation_pairs)}}
    return {"check": "4_mix_sweep", "passed": chosen is not None, "details": details}  # pass + every number
