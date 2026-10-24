# Tests of next-token choice: a seeded random generator makes it repeatable, the repetition penalty only touches used tokens
import torch  # tensors
from gpt2_demo_service.language_models.token_sampling import penalise_repeated_tokens, pick_next_token  # code under test


def test_same_seed_gives_same_token():  # seeded sampling is repeatable
    logits = torch.randn(1, 500, generator=torch.Generator().manual_seed(1))  # random scores
    first = pick_next_token(logits.clone(), 0.9, 40, 0.95, torch.Generator().manual_seed(5))  # first draw
    second = pick_next_token(logits.clone(), 0.9, 40, 0.95, torch.Generator().manual_seed(5))  # same seed
    assert first.item() == second.item()  # identical choice


def test_top_k_one_is_greedy():  # a single allowed token is always chosen
    logits = torch.tensor([[0.1, 3.0, 0.2, 2.9]])  # token 1 is best
    assert pick_next_token(logits, 1.0, 1, 1.0, torch.Generator().manual_seed(0)).item() == 1  # the maximum


def test_penalty_changes_only_used_tokens():  # other scores stay as they were
    logits = torch.tensor([[2.0, -2.0, 1.0, 1.0]])  # four scores
    penalised = penalise_repeated_tokens(logits.clone(), [0, 1], 2.0)  # tokens 0 and 1 were generated
    assert penalised.tolist() == [[1.0, -4.0, 1.0, 1.0]]  # positive halved, negative doubled, others untouched
    assert penalise_repeated_tokens(logits.clone(), [0], 1.0).tolist() == logits.tolist()  # penalty 1 does nothing
