# Tests that decoding with the cache gives the same scores as running the whole sequence, and that weight names did not change
import torch  # tensors
from gpt2_demo_service.language_models.decoding_cache import DecodingCache  # cache under test
from gpt2_demo_service.language_models.gpt_model import GPTModel  # network under test

LOGIT_TOLERANCE = 1e-5  # cached and full passes may differ by float rounding only


def make_cache_for(model: GPTModel, capacity: int) -> DecodingCache:  # empty cache shaped like the model
    attention = model.trf_blocks[0].att  # attention shape of the first layer
    return DecodingCache(len(model.trf_blocks), 1, attention.head_count, capacity, attention.head_dim, torch.float32)  # one request


def test_cached_steps_equal_full_forward(tiny_model):  # prefill then one token at a time reproduces the full pass at every position
    token_ids = torch.randint(0, 50257, (1, 24), generator=torch.Generator().manual_seed(7))  # a random sequence
    with torch.inference_mode():  # no gradients
        full_logits = tiny_model(token_ids)  # reference: the whole sequence at once
        decoding_cache = make_cache_for(tiny_model, 24)  # room for the whole sequence
        prefill_logits = tiny_model(token_ids[:, :16], decoding_cache)  # read the first 16 tokens in one go
        assert torch.allclose(prefill_logits, full_logits[:, :16], atol=LOGIT_TOLERANCE)  # prefill matches
        for position in range(16, 24):  # then one token per step
            step_logits = tiny_model(token_ids[:, position:position + 1], decoding_cache)  # one new position
            assert torch.allclose(step_logits[:, 0], full_logits[:, position], atol=LOGIT_TOLERANCE)  # same scores as the full pass


def test_last_position_only_returns_the_final_scores(tiny_model):  # the output head runs for one position
    token_ids = torch.randint(0, 50257, (1, 10), generator=torch.Generator().manual_seed(8))  # a random sequence
    with torch.inference_mode():  # no gradients
        full_logits = tiny_model(token_ids)  # scores at every position
        last_logits = tiny_model(token_ids, last_position_only=True)  # scores at the last position only
    assert last_logits.shape == (1, 1, 50257)  # one position kept
    assert torch.allclose(last_logits[:, 0], full_logits[:, -1], atol=LOGIT_TOLERANCE)  # same values


def test_weight_names_and_mask_buffer_are_unchanged(tiny_model):  # the saved weights of the live models must still load
    names = set(tiny_model.state_dict())  # every parameter and persistent buffer
    for expected_name in ("tok_emb.weight", "pos_emb.weight", "trf_blocks.0.att.W_query.weight", "trf_blocks.1.att.mask",
                          "trf_blocks.0.ff.layer.0.weight", "trf_blocks.0.norm1.scale", "final_norm.shift", "out_head.weight"):  # samples
        assert expected_name in names  # the name is still there
