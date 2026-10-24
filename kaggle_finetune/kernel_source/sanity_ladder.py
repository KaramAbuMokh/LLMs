# Karam's five checks before any full run (2026-10-24 02:08); each returns {"check", "passed", "details"}
import math  # finite checks
import statistics  # medians and means
import torch  # saving and reloading weights
from gpt_network import END_OF_TEXT_TOKEN_ID, GPTModel  # end token id 50256 and the network
from training_examples import IGNORED_LABEL, PROMPT_TEMPLATE, collate, make_targets  # format and labels
from training_loop import TrainingRun  # the same training code the full run uses
END_OF_TEXT = "<|endoftext|>"  # end token as text


def check_decoded_examples(train_pairs, train_examples, validation_pairs, tokenizer, settings):  # check 1: format, end token, masking
    details = []  # one record per decoded example
    for pair, example in list(zip(train_pairs, train_examples))[: settings.decoded_example_count]:  # first few training pairs
        expected_text = PROMPT_TEMPLATE.format(question=pair["q"]) + pair["a"] + END_OF_TEXT  # exact training format
        _, targets = make_targets(example, settings.mask_prompt_tokens)  # labels the loss sees
        scored_positions = [position for position, label in enumerate(targets) if label != IGNORED_LABEL]  # labels that count
        scored_text = tokenizer.decode([targets[position] for position in scored_positions])  # what the loss is on
        details.append({  # decoded text and each verdict
            "decoded_text": tokenizer.decode(example["tokens"]), "scored_label_text": scored_text,
            "text_matches_training_format": tokenizer.decode(example["tokens"]) == expected_text,  # prompt format right
            "same_ids_as_joint_tokenization": tokenizer.encode(expected_text, allowed_special={END_OF_TEXT}) == example["tokens"],
            "ends_with_exactly_one_end_token": example["tokens"][-1] == END_OF_TEXT_TOKEN_ID and example["tokens"].count(END_OF_TEXT_TOKEN_ID) == 1,
            "labels_are_answer_plus_end_token": scored_text == pair["a"] + END_OF_TEXT,  # prompt masked, answer + end scored
            "first_label_is_first_answer_token": scored_positions[0] == example["prompt_length"] - 1,  # mask ends at the right place
        })
    first_validation = validation_pairs[0]  # the dataset was written by the local prep script with the same template
    dataset_format_matches = PROMPT_TEMPLATE.format(question=first_validation["q"]) + first_validation["a"] + END_OF_TEXT == first_validation["training_text"]
    passed = dataset_format_matches and all(all(value for key, value in record.items() if key not in ("decoded_text", "scored_label_text"))
                                            for record in details)  # every verdict true
    return {"check": "1_decoded_examples", "passed": passed, "details": {"examples": details, "dataset_format_matches": dataset_format_matches}}


def check_starting_loss(metrics, settings):  # check 2: the current checkpoint scores as expected on the held-out passage
    answer_low, answer_high = settings.answer_loss_expected_range  # data review: about 3.0 on answer tokens
    legacy_low, legacy_high = settings.legacy_loss_expected_range  # Karam's all-token number: about 1.8
    passed = answer_low <= metrics["answer_loss"] <= answer_high and legacy_low <= metrics["legacy_full_text_loss"] <= legacy_high
    return {"check": "2_starting_validation_loss", "passed": passed, "details": metrics}  # pass + the numbers


def check_overfit_tiny_batch(model, train_examples, settings, device):  # check 3: one tiny batch must be memorised (loss near 0)
    inputs, targets = collate(train_examples[: settings.overfit_example_count], True, device)  # one fixed batch

    def same_batch_forever():  # the training loop asks for batches; it always gets this one
        while True:  # endless
            yield inputs, targets, "pairs", 0  # same batch every step

    run = TrainingRun(model, settings, device, settings.overfit_learning_rate, settings.overfit_steps, warmup_steps=10)  # short run
    losses = run.run(same_batch_forever())["train_loss"]  # loss per step
    passed = all(math.isfinite(loss) for loss in losses) and losses[-1] < settings.overfit_pass_loss  # fell close to zero
    details = {"examples": settings.overfit_example_count, "steps": len(losses), "loss_start": losses[0], "loss_end": losses[-1],
               "loss_every_25_steps": losses[::25]}  # the curve, thinned
    del run  # free the optimizer state
    return {"check": "3_overfit_tiny_batch", "passed": passed, "details": details}  # pass + the curve


def learning_rate_shape_is_right(rates, settings):  # warmup rises to the peak, then cosine never rises again
    peak_index = rates.index(max(rates))  # end of warmup
    rises = all(later >= earlier for earlier, later in zip(rates[:peak_index], rates[1:peak_index + 1]))  # warmup goes up
    falls = all(later <= earlier + 1e-12 for earlier, later in zip(rates[peak_index:], rates[peak_index + 1:]))  # decay goes down
    peak_right = abs(max(rates) - settings.peak_learning_rate) / settings.peak_learning_rate < 0.02  # reaches the peak rate
    floor = settings.peak_learning_rate * settings.minimum_learning_rate_ratio  # cosine target at the end
    end_right = abs(rates[-1] - floor) / floor < 0.10  # ends near the floor
    return rises and falls and peak_right and end_right, {"peak_step": peak_index + 1, "peak": max(rates), "last": rates[-1]}


def check_small_slice(history, starting_metrics, settings):  # check 4: losses fall, no NaN, sane gradients, warmup + cosine
    pair_losses = [loss for loss, kind in zip(history["train_loss"], history["batch_kind"]) if kind == "pairs"]  # answer-loss steps
    window = max(5, len(pair_losses) // 10)  # average over the first and last tenth
    train_start, train_end = statistics.mean(pair_losses[:window]), statistics.mean(pair_losses[-window:])  # smoothed ends
    validation_end = history["evaluations"][-1]["answer_loss"]  # after the slice
    skipped = set(history["overflow_skipped_steps"])  # fp16 calibration steps the scaler skipped (their norm is inf by design)
    norms = [norm for step, norm in zip(history["step"], history["gradient_norm"]) if step not in skipped]  # real update steps
    norms_sane = (all(math.isfinite(norm) for norm in norms) and max(norms) < settings.max_gradient_norm_sane
                  and len(skipped) <= 0.1 * len(history["step"]))  # no explosions, few skipped steps
    schedule_right, schedule = learning_rate_shape_is_right(history["learning_rate"], settings)  # warmup then cosine
    verdicts = {"no_nan": history["non_finite_losses"] == 0, "train_loss_fell": train_end < train_start,
                "validation_loss_fell": validation_end < starting_metrics["answer_loss"], "gradient_norms_sane": norms_sane,
                "learning_rate_warmup_then_cosine": schedule_right}  # each part of the check
    details = {**verdicts, "train_answer_loss_start": train_start, "train_answer_loss_end": train_end,
               "validation_answer_loss_start": starting_metrics["answer_loss"], "validation_answer_loss_end": validation_end,
               "validation_book_loss_start": starting_metrics["book_loss"], "validation_book_loss_end": history["evaluations"][-1]["book_loss"],
               "gradient_norm_median": statistics.median(norms), "gradient_norm_max": max(norms), "schedule": schedule,
               "fp16_overflow_skipped_steps": sorted(skipped),
               "steps": len(history["train_loss"])}  # numbers behind the verdicts
    return {"check": "4_small_slice", "passed": all(verdicts.values()), "details": details}  # pass + numbers


def check_generation_stops(answers, settings):  # check 5: answers from the slice model end with the end token
    stopped = sum(answer["stop_reason"] == "end_token" for answer in answers)  # answers that ended by themselves
    return {"check": "5_generation_stops_at_end_token", "passed": stopped >= settings.stop_pass_count,
            "details": {"stopped_with_end_token": stopped, "of": len(answers), "answers": answers}}  # pass + the answers


def check_save_and_reload(model, settings, device, output_folder):  # the saved file loads strictly into a fresh network, same outputs
    path = output_folder / "smoke_model.pth"  # temporary file
    torch.save(model.state_dict(), path)  # same call the full run uses for model.pth
    reloaded = GPTModel(settings.model_size, settings.dropout_rate)  # empty network (same parameter names as the Space's class)
    reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)  # every key must match
    reloaded.to(device).eval()  # same device, no dropout
    model.eval()  # no dropout
    probe = torch.randint(0, 50000, (1, 32), device=device)  # random token ids
    with torch.no_grad():  # just compare outputs
        same = torch.allclose(model(probe).float(), reloaded(probe).float(), atol=1e-4)  # identical weights give identical scores
    size_megabytes = path.stat().st_size / 1e6  # file size
    path.unlink()  # keep the smoke output small
    del reloaded  # free the copy
    return {"saved_megabytes": round(size_megabytes, 1), "strict_reload_same_outputs": same}  # result
