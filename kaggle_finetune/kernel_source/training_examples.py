# Turns the dataset files into token examples (exact training prompt + end token), batch plans and padded tensors
import json  # pair files
import os  # tokenizer cache location
import random  # shuffling
from pathlib import Path  # file paths
import torch  # tensors
from gpt_network import END_OF_TEXT_TOKEN_ID  # end token id 50256

IGNORED_LABEL = -100  # cross-entropy skips these positions
PROMPT_TEMPLATE = (  # the exact prompt Karam trained with (Drive notebook cell 32 and the dataset class)
    "Answer the following question with an appropriate response. "
    "Write a response that appropriately completes the qusetion."  # "qusetion" is spelled like this in training
    "\n\n### question:\n{question}\n\n### Response:\n"
)


def load_tokenizer(data_folder):  # GPT-2 tokenizer from the files shipped in the dataset (no download)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(data_folder)  # cache files sit at the dataset's top level
    import tiktoken  # imported after the cache folder is set
    return tiktoken.get_encoding("gpt2")  # the GPT-2 byte-pair encoding


def read_pairs(data_folder, file_name):  # one of the two pair files
    return json.loads((Path(data_folder) / file_name).read_text(encoding="utf-8"))  # list of pair records


def encode_pair(pair, tokenizer, settings):  # {"tokens", "prompt_length"} for one question-answer pair
    prompt_ids = tokenizer.encode_ordinary(PROMPT_TEMPLATE.format(question=pair["q"]))  # instruction + question + response header
    answer_ids = tokenizer.encode_ordinary(pair["a"])  # the answer
    tokens = (prompt_ids + answer_ids)[: settings.max_question_answer_tokens] + [END_OF_TEXT_TOKEN_ID]  # end token always last
    return {"tokens": tokens, "prompt_length": len(prompt_ids)}  # prompt length drives the label mask


def encode_book_windows(data_folder, file_name, tokenizer, settings):  # non-overlapping windows of 1024 + 1 tokens
    text = (Path(data_folder) / file_name).read_text(encoding="utf-8")  # book text
    token_ids = tokenizer.encode_ordinary(text)  # whole text as tokens
    step = settings.book_window_tokens  # window length
    return [{"tokens": token_ids[start:start + step + 1], "prompt_length": 0}  # +1 so inputs and targets both have 1024
            for start in range(0, len(token_ids) - step, step)]  # every full window


def make_targets(example, mask_prompt):  # input ids and label ids shifted by one
    tokens = example["tokens"]  # prompt + answer + end, or a book window
    inputs, targets = tokens[:-1], list(tokens[1:])  # predict the next token at every position
    if mask_prompt:  # loss only on the answer and the end token
        for position in range(max(0, example["prompt_length"] - 1)):  # targets that are still prompt tokens
            targets[position] = IGNORED_LABEL  # hide from the loss
    return inputs, targets  # unpadded lists


def collate(examples, mask_prompt, device):  # pads a batch to its longest example
    pairs = [make_targets(example, mask_prompt) for example in examples]  # inputs and targets per example
    longest = max(len(inputs) for inputs, _ in pairs)  # batch length
    input_rows = [inputs + [END_OF_TEXT_TOKEN_ID] * (longest - len(inputs)) for inputs, _ in pairs]  # pad inputs
    target_rows = [targets + [IGNORED_LABEL] * (longest - len(targets)) for _, targets in pairs]  # padding never counts
    to_tensor = lambda rows: torch.tensor(rows, dtype=torch.long, device=device)  # list of lists to a tensor
    return to_tensor(input_rows), to_tensor(target_rows)  # (batch, tokens) each


LENGTH_GROUP_BATCHES = 50  # pairs are sorted by length inside groups of 50 batches, so padding stays small


def plan_epoch(pair_examples, window_count, settings, seed):  # shuffled list of ("pairs" | "books", indices) batches
    shuffler = random.Random(seed)  # repeatable order per epoch
    batch_size = settings.question_answer_batch_size  # pairs per batch
    pair_order = list(range(len(pair_examples)))  # every training pair
    shuffler.shuffle(pair_order)  # new random order each epoch
    group_size = batch_size * LENGTH_GROUP_BATCHES  # pairs per length group
    batches = []  # the epoch plan
    for group_start in range(0, len(pair_order), group_size):  # one length group at a time
        group = sorted(pair_order[group_start:group_start + group_size], key=lambda index: len(pair_examples[index]["tokens"]))
        batches += [("pairs", group[start:start + batch_size]) for start in range(0, len(group) - batch_size + 1, batch_size)]
    window_order = list(range(window_count))  # every training book window
    shuffler.shuffle(window_order)  # random order
    window_order = window_order[: int(len(window_order) * settings.book_window_fraction)]  # optional share of the books
    book_size = settings.book_batch_size  # windows per batch
    batches += [("books", window_order[start:start + book_size]) for start in range(0, len(window_order) - book_size + 1, book_size)]
    shuffler.shuffle(batches)  # mix pair and book batches
    return batches  # one epoch


def stream_batches(pair_examples, window_examples, settings, device, mask_prompt=None):  # endless (inputs, targets, kind, epoch)
    mask = settings.mask_prompt_tokens if mask_prompt is None else mask_prompt  # answer-only loss by default
    epoch = 0  # epoch counter
    while True:  # the training loop decides when to stop
        for kind, indices in plan_epoch(pair_examples, len(window_examples), settings, settings.random_seed + epoch):  # one epoch
            if kind == "pairs":  # question-answer batch
                yield (*collate([pair_examples[index] for index in indices], mask, device), kind, epoch)  # answer-masked
            else:  # book windows
                yield (*collate([window_examples[index] for index in indices], False, device), kind, epoch)  # all tokens count
        epoch += 1  # next epoch, new order


def steps_per_epoch(pair_examples, window_count, settings):  # batches in one epoch (pair batches, book batches)
    plan = plan_epoch(pair_examples, window_count, settings, settings.random_seed)  # one epoch plan
    pair_batches = sum(kind == "pairs" for kind, _ in plan)  # question-answer batches
    return pair_batches, len(plan) - pair_batches  # pair batches, book batches
