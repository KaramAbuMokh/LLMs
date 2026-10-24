# Slow opt-in test on the real weights: 20 greedy tokens with the cache equal 20 greedy tokens re-reading the whole text each step
import os  # reads the weights location
from pathlib import Path  # weights path
import pytest  # slow marker and skip
import tiktoken  # GPT-2 tokenizer
import torch  # tensors
from gpt2_demo_service.language_models.got_text_generator import load_got_text_generator  # loads the 1.7 GB model

REAL_WEIGHTS_PATH = Path(os.environ.get("GOT_REAL_WEIGHTS_PATH", "../kaggle_finetune/mixed_run_results/model.pth"))  # the live model file
GREEDY_TOKEN_COUNT = 20  # tokens compared


@pytest.mark.slow
def test_cache_matches_full_rereading_on_real_weights():  # greedy decoding must pick the same tokens both ways
    if not REAL_WEIGHTS_PATH.is_file():  # nothing to test with
        pytest.skip("real weights not found at %s" % REAL_WEIGHTS_PATH)  # explain
    generator = load_got_text_generator(REAL_WEIGHTS_PATH, 600.0)  # real model
    prompt_ids = tiktoken.get_encoding("gpt2").encode("Answer the following question.\n\n### question:\nWho is Jon Snow?\n\n### Response:\n")  # short prompt
    with torch.inference_mode():  # no gradients
        reread_ids = list(prompt_ids)  # grows by one token per step
        for _ in range(GREEDY_TOKEN_COUNT):  # reference: the whole text every step
            reread_ids.append(int(generator.model(torch.tensor([reread_ids]))[0, -1].argmax()))  # best next token
        cache = generator.create_decoding_cache(len(prompt_ids) + GREEDY_TOKEN_COUNT)  # cached run
        logits = generator.model(torch.tensor([prompt_ids]), cache, last_position_only=True)[0, -1]  # prefill
        cached_ids = []  # greedy tokens with the cache
        for _ in range(GREEDY_TOKEN_COUNT):  # one token per step
            cached_ids.append(int(logits.argmax()))  # best next token
            logits = generator.model(torch.tensor([[cached_ids[-1]]]), cache, last_position_only=True)[0, -1]  # feed it back
    assert cached_ids == reread_ids[len(prompt_ids):]  # same 20 tokens
