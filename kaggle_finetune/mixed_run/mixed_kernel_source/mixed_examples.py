# Book text cut into 512-token blocks (the overlap with the block before is unscored context, so each token counts once per epoch),
# and the ONE shuffled training stream: every step holds a batch of question-answer pairs AND a batch of book blocks.
import json  # dataset files
import random  # shuffling the book blocks
from collections import Counter  # tokens per book
from pathlib import Path  # file paths
from gpt_network import END_OF_TEXT_TOKEN_ID  # end token id 50256 (never inside book blocks)
from training_examples import collate, plan_epoch  # padding and the morning run's length-grouped, shuffled pair batches


def read_book_parts(data_folder):  # [{"book", "part", "text"}] in a fixed order
    return json.loads((Path(data_folder) / "book_train_parts.json").read_text(encoding="utf-8"))  # six trimmed parts


def encode_book_blocks(parts, tokenizer, settings):  # every full block of every part; prompt_length marks the unscored overlap
    blocks, stride = [], settings.book_block_tokens - settings.book_overlap_tokens  # result, distance between block starts
    for part in parts:  # each book part on its own, so no block mixes two books
        token_ids = tokenizer.encode_ordinary(part["text"])  # the part as tokens
        for start in range(0, len(token_ids) - settings.book_block_tokens, stride):  # every block that fits
            context = 0 if start == 0 else settings.book_overlap_tokens + 1  # make_targets hides the first (prompt_length - 1) targets
            blocks.append({"tokens": token_ids[start:start + settings.book_block_tokens + 1], "prompt_length": context,
                           "book": part["book"], "part": part["part"], "start": start})  # +1 so inputs and targets both have 512
    return blocks  # about 5,600 blocks for the five books


def book_token_shares(blocks, settings):  # share of the scored book tokens per book (a part is a book or half of Feast)
    scored = Counter()  # book title -> scored tokens
    for block in blocks:  # every block
        scored[block["book"].split(" (")[0]] += settings.book_block_tokens - max(0, block["prompt_length"] - 1)  # tokens it scores
    total = sum(scored.values())  # all scored book tokens
    return {book: round(count / total, 4) for book, count in scored.items()}  # shares that add up to 1


def blocks_per_step(pair_examples, block_count, settings):  # book blocks per step so pairs and books finish an epoch together
    if settings.book_blocks_per_step:  # fixed by hand
        return settings.book_blocks_per_step  # as set
    pair_steps = len(plan_epoch(pair_examples, 0, settings, settings.random_seed))  # pair batches in one epoch (no book batches)
    return max(1, round(block_count / pair_steps))  # about 5 for the full data


def mixed_stream(pair_examples, block_examples, settings, device, blocks_each_step):  # endless ((x pairs, x books), (y pairs, y books), kind, epoch)
    block_shuffler = random.Random(settings.random_seed + 1000)  # own repeatable order for the books
    block_order, block_cursor, epoch = [], 0, 0  # current book pass, position in it, pair epoch
    while True:  # the training loop decides when to stop
        for _, pair_indices in plan_epoch(pair_examples, 0, settings, settings.random_seed + epoch):  # shuffled pair batches
            if block_cursor + blocks_each_step > len(block_order):  # this pass over the books is used up
                block_order, block_cursor = list(range(len(block_examples))), 0  # start a new pass
                block_shuffler.shuffle(block_order)  # in a new random order
            chosen = block_order[block_cursor:block_cursor + blocks_each_step]  # the next book blocks
            block_cursor += blocks_each_step  # move on
            pair_inputs, pair_targets = collate([pair_examples[i] for i in pair_indices], settings.mask_prompt_tokens, device)  # answer-only
            book_inputs, book_targets = collate([block_examples[i] for i in chosen], True, device)  # all tokens but the overlap
            yield (pair_inputs, book_inputs), (pair_targets, book_targets), "mixed", epoch  # one mixed step
        epoch += 1  # next pass over the pairs


def single_batch_forever(pair_examples, block_examples, settings, device):  # the same mixed batch every step (tiny over-fit check)
    pair_inputs, pair_targets = collate(pair_examples, settings.mask_prompt_tokens, device)  # fixed pairs
    book_inputs, book_targets = collate(block_examples, True, device)  # fixed book blocks
    while True:  # endless
        yield (pair_inputs, book_inputs), (pair_targets, book_targets), "mixed", 0  # same batch


def blocks_have_no_end_token(blocks):  # book text must never teach the model to stop in the middle of a story
    return all(END_OF_TEXT_TOKEN_ID not in block["tokens"] for block in blocks)  # True when clean
