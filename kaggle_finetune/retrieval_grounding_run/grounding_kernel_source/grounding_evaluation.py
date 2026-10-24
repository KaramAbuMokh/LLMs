# Validation numbers of the grounding run: the answer loss with what retrieval returns (drives early stopping), with the oracle passage, without any evidence, and the book voice
from model_evaluation import mean_loss  # token-weighted mean loss over examples


def grounding_validation_losses(model, data, settings, device):  # the four numbers of one validation pass
    retrieved = mean_loss(model, data["retrieved_validation"], settings.grounded_batch_size, True, settings, device)  # development questions with the Space's evidence
    oracle = mean_loss(model, data["oracle_validation"], settings.grounded_batch_size, True, settings, device)  # held-out pairs with their answer passage
    plain = mean_loss(model, data["plain_validation"][:settings.validation_plain_pair_count], settings.question_answer_batch_size, True, settings, device)  # no evidence at all
    book = mean_loss(model, data["validation_windows"][:settings.validation_window_count], settings.book_batch_size, False, settings, device)  # held-out book passage
    return {"answer_loss": retrieved, "oracle_answer_loss": oracle, "plain_answer_loss": plain, "book_loss": book}  # answer_loss drives early stopping and best-weights saving


def mean_seconds_by_kind(history, skipped_steps=5):  # seconds per step for each batch kind, warm-up steps left out
    timed = list(zip(history["batch_kind"], history["step_seconds"]))[skipped_steps:]  # steps after the warm-up
    kinds = {kind for kind, _ in timed}  # kinds seen
    return {kind: sum(seconds for other, seconds in timed if other == kind) / sum(other == kind for other, _ in timed) for kind in kinds}  # mean per kind
