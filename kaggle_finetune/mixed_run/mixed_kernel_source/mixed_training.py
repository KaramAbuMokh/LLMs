# One optimizer step on a pair batch AND a book batch: loss = weight x pair loss + (1 - weight) x book loss.
# Everything else (AdamW, warmup + cosine, fp16 scaling, clipping, evaluations, early stopping, saving) is the morning run's TrainingRun.
import torch  # gradient clipping
from model_evaluation import autocast_context  # fp16 switch
from training_loop import TrainingRun, language_model_loss  # the morning run's training session and its memory-saving loss


class MixedTrainingRun(TrainingRun):  # same session, but each batch is a (pairs, books) pair of batches
    def __init__(self, model, settings, device, learning_rate, total_steps, warmup_steps, question_answer_weight):
        super().__init__(model, settings, device, learning_rate, total_steps, warmup_steps)  # optimizer, schedule, scaler, history
        self.question_answer_weight = question_answer_weight  # share of the loss that comes from the pairs
        self.history.update({"pair_loss": [], "book_loss": [], "question_answer_weight": question_answer_weight})  # per-source losses

    def weighted_backward(self, inputs, targets, weight):  # forward one batch, backward its weighted loss; returns the plain loss
        with autocast_context(self.settings, self.device):  # fp16 forward on the T4
            loss = language_model_loss(self.model, inputs, targets)  # mean loss over the tokens that count
        self.scaler.scale(loss * weight).backward()  # gradients add up across the two batches
        return loss.item()  # plain number for the history

    def train_one_batch(self, inputs, targets):  # pairs then books, one update; returns combined loss, pre-clip norm, skipped flag
        self.model.train()  # dropout on
        self.optimizer.zero_grad(set_to_none=True)  # clear old gradients
        pair_loss = self.weighted_backward(inputs[0], targets[0], self.question_answer_weight)  # question-answer part
        book_loss = self.weighted_backward(inputs[1], targets[1], 1.0 - self.question_answer_weight)  # book-text part
        self.scaler.unscale_(self.optimizer)  # real gradient sizes before clipping
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.gradient_clip_norm)  # clip to 1.0
        scale_before = self.scaler.get_scale()  # fp16 loss scale used for this step
        self.scaler.step(self.optimizer)  # update (skipped by the scaler if gradients overflowed)
        self.scaler.update()  # adjust the loss scale
        self.scheduler.step()  # next learning rate
        self.history["pair_loss"].append(pair_loss)  # question-answer training loss of this step
        self.history["book_loss"].append(book_loss)  # book training loss of this step
        combined = self.question_answer_weight * pair_loss + (1.0 - self.question_answer_weight) * book_loss  # what was minimised
        return combined, float(gradient_norm), self.scaler.get_scale() < scale_before  # scale cut = this step was skipped
