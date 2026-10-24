# Validation loss / perplexity and answers to the fixed questions, used the same way for the current and the new model
import math  # perplexity
import torch  # tensors
import torch.nn.functional as functional  # cross-entropy
from gpt_network import CONTEXT_LENGTH, END_OF_TEXT_TOKEN_ID  # model limits and end token
from training_examples import IGNORED_LABEL, PROMPT_TEMPLATE, collate  # padding and prompt
NEXT_QUESTION_MARKER = "###"  # the model starting a new "### question" block means it failed to stop


def autocast_context(settings, device):  # fp16 on the GPU, plain fp32 on the CPU
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=settings.use_fp16 and device.type == "cuda")  # mixed precision


def token_loss_sums(model, inputs, targets, settings, device):  # summed loss and counted tokens for one batch
    with autocast_context(settings, device):  # fast maths on the T4
        logits = model(inputs)  # (batch, tokens, vocabulary)
    loss_sum = functional.cross_entropy(  # summed over the tokens that count
        logits.float().flatten(0, 1), targets.flatten(), ignore_index=IGNORED_LABEL, reduction="sum",
    )
    return loss_sum.item(), int((targets != IGNORED_LABEL).sum().item())  # plain numbers


@torch.no_grad()  # scoring only: no gradients, less memory
def mean_loss(model, examples, batch_size, mask_prompt, settings, device):  # token-weighted mean loss over a list of examples
    was_training = model.training  # restore the mode afterwards
    model.eval()  # no dropout
    total_loss, total_tokens = 0.0, 0  # running sums
    for start in range(0, len(examples), batch_size):  # every batch, in a fixed order
        inputs, targets = collate(examples[start:start + batch_size], mask_prompt, device)  # padded tensors
        batch_loss, batch_tokens = token_loss_sums(model, inputs, targets, settings, device)  # loss of this batch
        total_loss, total_tokens = total_loss + batch_loss, total_tokens + batch_tokens  # add up
    model.train(was_training)  # back to the earlier mode
    return total_loss / max(total_tokens, 1)  # mean loss per counted token


def validation_metrics(model, validation_pairs, validation_windows, settings, device):  # every number the comparison needs
    answer_loss = mean_loss(model, validation_pairs, settings.question_answer_batch_size, True, settings, device)  # answer + end token
    legacy_loss = mean_loss(model, validation_pairs, settings.question_answer_batch_size, False, settings, device)  # all tokens, like Karam
    book_loss = mean_loss(model, validation_windows, settings.book_batch_size, False, settings, device)  # held-out book passage
    return {  # losses and perplexities (perplexity = e to the loss)
        "answer_loss": answer_loss, "answer_perplexity": math.exp(answer_loss),
        "legacy_full_text_loss": legacy_loss, "legacy_full_text_perplexity": math.exp(legacy_loss),
        "book_loss": book_loss, "book_perplexity": math.exp(book_loss),
    }


def pick_next_token(logits, generated_ids, settings, generator):  # repetition penalty, temperature, top-k, top-p, then sample
    logits = logits.float().clone()  # (vocabulary,), full precision
    for token_id in set(generated_ids):  # tokens already used in the answer
        # shrink positive scores and push negative ones further down, so used tokens become less likely
        logits[token_id] = logits[token_id] / settings.repetition_penalty if logits[token_id] > 0 else logits[token_id] * settings.repetition_penalty
    logits = logits / settings.temperature  # sharpen or flatten
    top_values, top_indices = torch.topk(logits, settings.top_k)  # keep the k best tokens
    probabilities = torch.softmax(top_values, dim=-1)  # their probabilities
    cumulative = torch.cumsum(probabilities, dim=-1)  # running total, best first
    keep = cumulative - probabilities < settings.top_p  # smallest set whose total reaches top_p
    probabilities = probabilities * keep  # drop the tail
    choice = torch.multinomial(probabilities / probabilities.sum(), 1, generator=generator)  # sample one
    return int(top_indices[choice].item())  # chosen token id


@torch.no_grad()  # generation only: no gradients
def answer_question(model, tokenizer, question, seed, settings, device):  # one answer and why it stopped
    model.eval()  # no dropout
    generator = torch.Generator(device="cpu").manual_seed(seed)  # same seed for both models
    token_ids = tokenizer.encode_ordinary(PROMPT_TEMPLATE.format(question=question))  # exact training prompt
    generated_ids, stop_reason = [], "max_tokens"  # answer tokens, default stop reason
    for _ in range(settings.max_new_tokens):  # one token per round (no cache, fine for 120 tokens)
        context = torch.tensor([(token_ids + generated_ids)[-CONTEXT_LENGTH:]], device=device)  # prompt + answer so far
        with autocast_context(settings, device):  # fast maths on the T4
            logits = model(context)[0, -1].cpu()  # scores for the next token
        next_id = pick_next_token(logits, generated_ids, settings, generator)  # sampled token
        if next_id == END_OF_TEXT_TOKEN_ID:  # the model says it is done
            stop_reason = "end_token"  # the behaviour we want
            break  # stop
        generated_ids.append(next_id)  # keep the token
        if NEXT_QUESTION_MARKER in tokenizer.decode(generated_ids):  # it started inventing a new question
            stop_reason = "next_question_marker"  # did not stop by itself
            break  # stop
    answer_text = tokenizer.decode(generated_ids).split(NEXT_QUESTION_MARKER)[0].strip()  # clean answer
    return {"question": question, "answer": answer_text, "stop_reason": stop_reason, "answer_tokens": len(generated_ids)}


def answer_fixed_questions(model, tokenizer, questions, settings, device):  # answers to all fixed questions
    return [answer_question(model, tokenizer, question, settings.generation_seed + number, settings, device)  # seed per question
            for number, question in enumerate(questions)]  # in a fixed order
