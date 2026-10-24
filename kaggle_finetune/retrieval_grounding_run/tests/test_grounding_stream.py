# Tests of the 70/15/15 training stream: the share of each kind, the answer-only labels and the batch shapes
import torch  # tensors
from grounding_settings import build_grounding_settings  # dry-run numbers
from grounding_stream import grounding_stream, plan_grounding_epoch  # code under test
from training_examples import IGNORED_LABEL  # label that hides prompt tokens


def make_examples(count: int, prompt_length: int) -> list[dict]:  # invented token sequences
    return [{"tokens": list(range(1, 30 + number)), "prompt_length": prompt_length} for number in range(count)]  # growing lengths


def test_epoch_plan_has_the_requested_shares() -> None:  # about 21 plain and 21 book steps per 100 grounded steps
    settings = build_grounding_settings("grounding_full")  # production numbers
    plan = plan_grounding_epoch(make_examples(4000, 10), make_examples(2000, 5), make_examples(500, 3), settings, seed=1)  # 1000 grounded steps
    kinds = [kind for kind, _ in plan]  # the order of kinds
    assert kinds.count("grounded") == 1000 and kinds.count("plain") == 210 and kinds.count("books") == 210  # 70/15/15 by steps
    assert all(len(indices) == 4 for kind, indices in plan if kind == "grounded")  # grounded batches are full


def test_stream_hides_the_prompt_and_pads() -> None:  # labels before the answer are ignored; padding is ignored too
    settings = build_grounding_settings("dry_grounding")  # tiny numbers
    stream = grounding_stream(make_examples(8, 10), make_examples(8, 5), make_examples(8, 3), settings, torch.device("cpu"))  # all three kinds
    seen = {}  # first batch of each kind
    for _ in range(12):  # enough steps to see every kind
        inputs, targets, kind, epoch = next(stream)  # one step
        seen.setdefault(kind, (inputs, targets))  # remember the first of its kind
    assert set(seen) == {"grounded", "plain", "books"}  # every kind appeared
    inputs, targets = seen["grounded"]  # a grounded batch
    assert inputs.shape == targets.shape and (targets[:, :9] == IGNORED_LABEL).all()  # the first prompt_length - 1 targets are hidden
    assert (targets[:, 9] != IGNORED_LABEL).any()  # the first answer token counts
