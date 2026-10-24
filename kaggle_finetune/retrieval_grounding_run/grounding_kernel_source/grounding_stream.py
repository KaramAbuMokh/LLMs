# The ONE shuffled training stream of the grounding run: grounded sequences (70 percent), plain Q&A pairs (15) and book blocks (15), one kind per step
import random  # shuffling
from training_examples import collate  # padding and answer-only labels


def plan_grounding_epoch(grounded, plain_pairs, blocks, settings, seed):  # shuffled list of (kind, example indices) steps
    shuffler = random.Random(seed)  # repeatable order per epoch
    grounded_order = list(range(len(grounded)))  # every grounded sequence
    shuffler.shuffle(grounded_order)  # random order
    group_size = settings.grounded_batch_size * 50  # sequences per length group
    steps = []  # the epoch plan
    for group_start in range(0, len(grounded_order), group_size):  # sort by length inside groups, so padding stays small
        group = sorted(grounded_order[group_start:group_start + group_size], key=lambda index: len(grounded[index]["tokens"]))  # short to long
        steps += [("grounded", group[start:start + settings.grounded_batch_size]) for start in range(0, len(group) - settings.grounded_batch_size + 1, settings.grounded_batch_size)]  # full grounded batches of this group
    grounded_steps = len(steps)  # grounded steps in this epoch
    plain_count = round(grounded_steps * settings.plain_batches_per_hundred_grounded / 100)  # plain steps
    book_count = round(grounded_steps * settings.book_batches_per_hundred_grounded / 100)  # book steps
    plain_order = list(range(len(plain_pairs)))  # every plain pair
    shuffler.shuffle(plain_order)  # random order
    batch_size = settings.question_answer_batch_size  # pairs per plain step
    steps += [("plain", [plain_order[(number * batch_size + offset) % len(plain_order)] for offset in range(batch_size)]) for number in range(plain_count)]  # wraps if the pairs run out
    block_order = list(range(len(blocks)))  # every book block
    shuffler.shuffle(block_order)  # random order
    per_batch = settings.book_blocks_per_batch  # blocks per book step
    steps += [("books", [block_order[(number * per_batch + offset) % len(block_order)] for offset in range(per_batch)]) for number in range(book_count)]  # wraps too
    shuffler.shuffle(steps)  # mix the three kinds
    return steps  # one epoch


def grounding_stream(grounded, plain_pairs, blocks, settings, device):  # endless (inputs, targets, kind, epoch)
    sources = {"grounded": grounded, "plain": plain_pairs, "books": blocks}  # example lists by kind
    epoch = 0  # epoch counter
    while True:  # the training loop decides when to stop
        for kind, indices in plan_grounding_epoch(grounded, plain_pairs, blocks, settings, settings.random_seed + epoch):  # one epoch
            inputs, targets = collate([sources[kind][index] for index in indices], True, device)  # answer-only labels (book overlap is unscored context)
            yield inputs, targets, kind, epoch  # one step
        epoch += 1  # next epoch, new order


def count_epoch_steps(grounded, plain_pairs, blocks, settings):  # steps in one epoch
    return len(plan_grounding_epoch(grounded, plain_pairs, blocks, settings, settings.random_seed))  # length of the plan
