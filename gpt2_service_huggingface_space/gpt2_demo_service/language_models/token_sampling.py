# Picks the next token from the model's scores: repetition penalty, temperature, top-k and top-p (nucleus) sampling
import torch  # tensor library


def penalise_repeated_tokens(logits, repeated_token_ids, penalty):  # makes already-used tokens less likely
    if penalty == 1.0 or not repeated_token_ids:  # nothing to do
        return logits  # scores unchanged
    token_ids = torch.tensor(sorted(set(repeated_token_ids)))  # each used token once
    scores = logits[0, token_ids]  # their current scores
    logits[0, token_ids] = torch.where(scores > 0, scores / penalty, scores * penalty)  # push scores toward "unlikely"
    return logits  # penalised scores


def pick_next_token(logits, temperature, top_k, top_p, random_generator: torch.Generator | None = None):  # returns the chosen token id as a (1, 1) tensor
    logits = logits / temperature  # lower temperature = safer choices
    best_scores, _ = torch.topk(logits, min(top_k, logits.shape[-1]))  # the top_k best scores
    logits = torch.where(logits < best_scores[:, -1:], torch.tensor(float("-inf")), logits)  # drop all others
    probabilities = torch.softmax(logits, dim=-1)  # scores to chances
    sorted_probabilities, sorted_ids = torch.sort(probabilities, descending=True)  # likeliest first
    chance_before_token = torch.cumsum(sorted_probabilities, dim=-1) - sorted_probabilities  # chance covered by likelier tokens
    sorted_probabilities = sorted_probabilities.masked_fill(chance_before_token > top_p, 0.0)  # keep the top_p nucleus only
    chosen_position = torch.multinomial(sorted_probabilities, num_samples=1, generator=random_generator)  # random pick inside the nucleus
    return sorted_ids.gather(-1, chosen_position)  # position back to token id
