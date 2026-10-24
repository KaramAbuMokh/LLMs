# Continued training: AdamW, warmup then cosine decay, gradient clipping, fp16, evaluations, early stopping and best-checkpoint saving
import math  # cosine curve and finite checks
import time  # time limits and step timing
import torch  # tensors and optimizer
import torch.nn.functional as functional  # cross-entropy
from torch.utils.checkpoint import checkpoint  # recompute the word scores of a chunk in backward
from model_evaluation import autocast_context  # fp16 switch
from training_examples import IGNORED_LABEL  # padding label


def learning_rate_multiplier(step, total_steps, warmup_steps, minimum_ratio):  # share of the peak rate at a step
    if step < warmup_steps:  # warmup: straight ramp up
        return (step + 1) / warmup_steps  # reaches 1.0 at the end of warmup
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))  # 0 -> 1 over the decay
    return minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))  # cosine from 1.0 down to the minimum


def chunk_loss_sum(hidden_chunk, target_chunk, head_weight):  # summed loss of one chunk of tokens
    logits = functional.linear(hidden_chunk, head_weight)  # word scores for these tokens only
    return functional.cross_entropy(logits.float(), target_chunk, ignore_index=IGNORED_LABEL, reduction="sum")  # fp32 loss


def language_model_loss(model, inputs, targets, chunk_tokens=1024):  # mean next-token loss without holding all word scores at once
    hidden = model.hidden_states(inputs).flatten(0, 1)  # (batch x tokens, width)
    flat_targets = targets.flatten()  # matching labels
    total = hidden.new_zeros((), dtype=torch.float32)  # running loss sum
    for start in range(0, flat_targets.shape[0], chunk_tokens):  # 1024 tokens at a time (about 0.2 GB of scores, not 2.5 GB)
        target_chunk = flat_targets[start:start + chunk_tokens]  # labels of this chunk
        if (target_chunk != IGNORED_LABEL).any():  # skip chunks that are all prompt or padding
            total = total + checkpoint(chunk_loss_sum, hidden[start:start + chunk_tokens], target_chunk,
                                       model.out_head.weight, use_reentrant=False)  # scores rebuilt in backward
    return total / (flat_targets != IGNORED_LABEL).sum().clamp(min=1)  # mean over counted tokens


def build_optimizer(model, learning_rate, settings, device):  # AdamW without weight decay on biases and norm parameters
    decayed = [parameter for parameter in model.parameters() if parameter.dim() >= 2]  # weight matrices and embeddings
    not_decayed = [parameter for parameter in model.parameters() if parameter.dim() < 2]  # biases, scales, shifts
    groups = [{"params": decayed, "weight_decay": settings.weight_decay}, {"params": not_decayed, "weight_decay": 0.0}]  # two groups
    return torch.optim.AdamW(groups, lr=learning_rate, fused=device.type == "cuda")  # fused kernel on the GPU


class TrainingRun:  # one training session with its own optimizer, schedule and history
    def __init__(self, model, settings, device, learning_rate, total_steps, warmup_steps):  # sets up the optimizer and schedule
        self.model, self.settings, self.device, self.total_steps = model, settings, device, total_steps  # shared objects
        self.optimizer = build_optimizer(model, learning_rate, settings, device)  # AdamW
        ratio = settings.minimum_learning_rate_ratio  # floor of the cosine
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(  # per-step learning rate
            self.optimizer, lambda step: learning_rate_multiplier(step, total_steps, warmup_steps, ratio))
        self.scaler = torch.amp.GradScaler("cuda", enabled=settings.use_fp16 and device.type == "cuda")  # fp16 loss scaling
        self.history = {"step": [], "train_loss": [], "gradient_norm": [], "learning_rate": [], "batch_kind": [],
                        "step_seconds": [], "evaluations": [], "non_finite_losses": 0, "overflow_skipped_steps": [],
                        "stopped_because": "step_limit"}  # overflow_skipped_steps = fp16 steps the scaler skipped (normal at the start)

    def train_one_batch(self, inputs, targets):  # forward, backward, clip, step; returns loss, pre-clip gradient norm, skipped flag
        self.model.train()  # dropout on
        with autocast_context(self.settings, self.device):  # fp16 forward
            loss = language_model_loss(self.model, inputs, targets)  # mean loss on the tokens that count
        self.optimizer.zero_grad(set_to_none=True)  # clear old gradients
        self.scaler.scale(loss).backward()  # gradients (scaled for fp16)
        self.scaler.unscale_(self.optimizer)  # real gradient sizes before clipping
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.gradient_clip_norm)  # clip to 1.0
        scale_before = self.scaler.get_scale()  # fp16 loss scale used for this step
        self.scaler.step(self.optimizer)  # update (skipped by the scaler if gradients overflowed)
        self.scaler.update()  # adjust the loss scale
        self.scheduler.step()  # next learning rate
        return loss.item(), float(gradient_norm), self.scaler.get_scale() < scale_before  # scale cut = this step was skipped

    def record_step(self, step, loss, gradient_norm, learning_rate, kind, seconds):  # one line of history
        for key, value in (("step", step), ("train_loss", loss), ("gradient_norm", gradient_norm),
                           ("learning_rate", learning_rate), ("batch_kind", kind), ("step_seconds", seconds)):  # each field
            self.history[key].append(value)  # append to its list
        if not math.isfinite(loss):  # NaN or infinite loss
            self.history["non_finite_losses"] += 1  # count it

    def run(self, batch_stream, evaluate=None, evaluate_every=0, save_best=None, patience=None, max_seconds=None, starting_best_loss=math.inf):
        """Trains up to total_steps; evaluate() returns a dict with "answer_loss"; save_best(model) runs on each new best.
        starting_best_loss = the current model's score, so only real improvements over it are saved."""
        best_loss, evaluations_without_improvement, started = starting_best_loss, 0, time.time()  # early-stopping state
        for step in range(1, self.total_steps + 1):  # step counter starts at 1
            inputs, targets, kind, epoch = next(batch_stream)  # next batch
            learning_rate = self.scheduler.get_last_lr()[0]  # rate used for this step
            step_started = time.time()  # timing
            loss, gradient_norm, skipped = self.train_one_batch(inputs, targets)  # one update
            if skipped:  # fp16 overflow: the scaler skipped the update and lowered the scale
                self.history["overflow_skipped_steps"].append(step)  # keep the step number
            if self.device.type == "cuda":  # wait for the GPU so the timing is real
                torch.cuda.synchronize()  # finish queued work
            self.record_step(step, loss, gradient_norm, learning_rate, kind, time.time() - step_started)  # history
            if evaluate and (step % evaluate_every == 0 or step == self.total_steps):  # time for a validation pass
                metrics = {"step": step, "epoch": epoch, **evaluate(self.model)}  # validation numbers
                self.history["evaluations"].append(metrics)  # keep them
                print(f"step {step} epoch {epoch} train {loss:.3f} val answer {metrics['answer_loss']:.4f}", flush=True)  # log
                if metrics["answer_loss"] < best_loss - self.settings.minimum_improvement:  # real improvement
                    best_loss, evaluations_without_improvement = metrics["answer_loss"], 0  # new best
                    if save_best:  # full run keeps the best weights
                        save_best(self.model)  # write model.pth
                else:  # no improvement
                    evaluations_without_improvement += 1  # count towards patience
                if patience and evaluations_without_improvement >= patience:  # early stopping
                    self.history["stopped_because"] = f"early_stopping_at_step_{step}"  # reason
                    break  # stop training
            if max_seconds and time.time() - started > max_seconds:  # time limit for the session
                self.history["stopped_because"] = f"time_limit_at_step_{step}"  # reason
                break  # stop training
        self.history["best_answer_loss"] = best_loss  # best validation answer loss seen
        self.history["training_seconds"] = time.time() - started  # wall time
        return self.history  # everything that happened


def save_history_plots(history, output_folder, name_prefix):  # learning-rate and loss curves as PNG files
    import matplotlib  # plotting library on Kaggle
    matplotlib.use("Agg")  # no screen on Kaggle
    import matplotlib.pyplot as pyplot  # figure drawing
    figure, (rate_axis, loss_axis) = pyplot.subplots(1, 2, figsize=(11, 4))  # two panels side by side
    rate_axis.plot(history["step"], history["learning_rate"])  # warmup then cosine
    rate_axis.set_title("learning rate per step")  # panel title
    loss_axis.plot(history["step"], history["train_loss"], alpha=0.4, label="train (per batch)")  # noisy training loss
    evaluations = history["evaluations"]  # validation points
    loss_axis.plot([record["step"] for record in evaluations], [record["answer_loss"] for record in evaluations], "o-", label="validation answer loss")
    loss_axis.set_title("loss")  # panel title
    loss_axis.legend()  # which line is which
    figure.tight_layout()  # no overlapping labels
    figure.savefig(output_folder / f"{name_prefix}_curves.png", dpi=110)  # write the PNG
    pyplot.close(figure)  # free memory


def estimate_full_run(history, train_examples, window_count, settings, evaluation_seconds, steps_per_epoch):  # hours from measured steps
    timed = list(zip(history["batch_kind"], history["step_seconds"]))[10:]  # skip warm-up steps
    seconds = {kind: sum(s for k, s in timed if k == kind) / max(1, sum(k == kind for k, _ in timed)) for kind in ("pairs", "books")}
    pair_batches, book_batches = steps_per_epoch(train_examples, window_count, settings)  # full epoch size
    evaluations = (pair_batches + book_batches) / settings.evaluation_every_steps  # validation passes per epoch
    epoch_hours = (pair_batches * seconds["pairs"] + book_batches * seconds["books"] + evaluations * (evaluation_seconds + 10)) / 3600
    return {"seconds_per_pair_batch": seconds["pairs"], "seconds_per_book_batch": seconds["books"], "pair_batches_per_epoch": pair_batches,
            "book_batches_per_epoch": book_batches, "hours_per_epoch": epoch_hours, "hours_for_max_epochs": epoch_hours * settings.max_epochs}
