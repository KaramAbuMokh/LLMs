# Scores a model the same way for all three (original, this morning's, mixed): held-out losses, stop rate and the facts score.
# Sampling uses the Space's settings (temperature 0.7, top-k 40, top-p 0.9, repetition penalty 1.1, 120 tokens, stop at "###").
import time  # scoring time
from fact_questions import FACT_QUESTIONS, score_fact_answers  # the 30 questions and their rules
from model_evaluation import answer_question, mean_loss, validation_metrics  # the morning run's scoring and sampling


def validation_losses(model, answerable_pairs, windows, settings, device):  # quick numbers for each check during training
    answer_loss = mean_loss(model, answerable_pairs, settings.question_answer_batch_size, True, settings, device)  # clean held-out pairs
    book_loss = mean_loss(model, windows, settings.book_batch_size, False, settings, device)  # held-out book passage
    return {"answer_loss": answer_loss, "book_loss": book_loss}  # answer_loss drives early stopping and best-weights saving


def sample_fact_answers(model, tokenizer, settings, device, question_limit=None):  # each fact question sampled 3 times
    sampled = []  # one record per question
    for number, (question, _, _) in enumerate(FACT_QUESTIONS[:question_limit]):  # in a fixed order
        seeds = [settings.generation_seed + 100 * number + sample for sample in range(settings.fact_samples_per_question)]  # same for all models
        samples = [answer_question(model, tokenizer, question, seed, settings, device) for seed in seeds]  # sampled answers
        sampled.append({"question": question, "samples": [{key: sample[key] for key in ("answer", "stop_reason", "answer_tokens")}
                                                          for sample in samples]})  # keep the useful fields
    return sampled  # answers, not yet scored


def stop_counts(sampled, fixed_count=10):  # how often the 10 fixed questions ended with the end token
    fixed = sampled[:fixed_count]  # the fixed questions come first
    stopped = [sum(sample["stop_reason"] == "end_token" for sample in record["samples"]) for record in fixed]  # per question
    return {"stop_samples": sum(stopped), "stop_samples_of": sum(len(record["samples"]) for record in fixed),
            "stop_majority": sum(count * 2 > len(record["samples"]) for count, record in zip(stopped, fixed)), "stop_majority_of": len(fixed)}


def evaluate_model(model, data, tokenizer, settings, device):  # every number of the three-model table for one model
    started = time.time()  # timing
    metrics = validation_metrics(model, data["validation_examples"], data["validation_windows"], settings, device)  # all 1,233 + book
    metrics["clean_answer_loss"] = mean_loss(model, data["answerable_validation_examples"], settings.question_answer_batch_size,
                                             True, settings, device)  # the 816 answerable held-out pairs
    sampled = sample_fact_answers(model, tokenizer, settings, device)  # 30 questions x 3 samples
    facts = score_fact_answers(sampled)  # marks every sample right or wrong
    return {"metrics": metrics, **facts, **stop_counts(sampled), "fact_answers": sampled, "scoring_seconds": time.time() - started}
