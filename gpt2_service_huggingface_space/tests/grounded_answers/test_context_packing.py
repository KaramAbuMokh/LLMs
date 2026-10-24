# Tests of the prompt packer: property tests over random budgets and evidence, plus the overlap, trimming and layout rules
import random  # seeded random trials
import pytest  # parametrisation
from gpt2_demo_service.grounded_answers.answer_settings import PromptLayout  # layout names
from gpt2_demo_service.grounded_answers.context_packing import MINIMUM_TRIMMED_TOKENS, PromptBudget, count_tokens, pack_prompt, trim_to_sentence_budget  # code under test
from gpt2_demo_service.grounded_answers.prompt_layouts import render_question_block  # the question block text

CONTEXT_LENGTH = 1024  # production context window
TRIAL_COUNT = 150  # random trials in the property test


def test_trim_returns_a_sentence_aligned_prefix_that_fits(evidence_factory, gpt2_tokenizer):  # never longer than asked, never mid-sentence
    text = evidence_factory["text"](random.Random(1), 12)  # twelve invented sentences
    assert trim_to_sentence_budget(text, 10_000, gpt2_tokenizer) == text  # fits whole: unchanged
    for limit in (0, 5, 40, 80, 120):  # several limits
        trimmed = trim_to_sentence_budget(text, limit, gpt2_tokenizer)  # cut
        assert text.startswith(trimmed) and count_tokens(trimmed, gpt2_tokenizer) <= limit  # a prefix that fits
        assert trimmed == "" or trimmed.endswith(".")  # ends at a sentence end (or is empty)
    assert trim_to_sentence_budget(text, 80, gpt2_tokenizer) != text and len(trim_to_sentence_budget(text, 80, gpt2_tokenizer)) > 0  # a real cut


@pytest.mark.parametrize("seed", range(TRIAL_COUNT))
def test_packer_invariants_over_random_inputs(seed, evidence_factory, gpt2_tokenizer):  # the hard limits hold whatever the input
    random_source = random.Random(seed)  # one trial
    question = evidence_factory["text"](random_source, random_source.randint(1, 4))  # a question of 1 to 4 sentences
    passages = [evidence_factory["passage"]("bp-%d" % number, evidence_factory["text"](random_source, random_source.randint(2, 14)), 0, number * 100_000)
                for number in range(random_source.randint(0, 3))]  # up to three passages, far apart so none overlap
    pairs = [evidence_factory["pair"]("qa-%d" % number, evidence_factory["sentence"](random_source), evidence_factory["text"](random_source, random_source.randint(1, 6)))
             for number in range(random_source.randint(0, 2))]  # up to two examples
    layout = random_source.choice(list(PromptLayout))  # any layout
    budget = PromptBudget(CONTEXT_LENGTH, random_source.randint(120, 200), random_source.randint(0, 800), 8)  # random limits
    packed = pack_prompt(question, evidence_factory["evidence"](passages, pairs), layout, budget, gpt2_tokenizer)  # pack
    assert len(packed.prompt_token_ids) + budget.answer_reserve_tokens + budget.safety_margin_tokens <= CONTEXT_LENGTH  # never over the window
    base_tokens = count_tokens(render_question_block(question), gpt2_tokenizer)  # the question alone
    assert len(packed.prompt_token_ids) - base_tokens <= budget.context_token_budget  # retrieved text within its budget
    assert packed.prompt_text.endswith(render_question_block(question))  # the question block is intact and last
    assert tuple(packed.prompt_token_ids) == tuple(gpt2_tokenizer.encode(packed.prompt_text, disallowed_special=()))  # ids match the text
    assert packed.passages == tuple(passages[:len(packed.passages)]) and packed.pair_examples == tuple(pairs[:len(packed.pair_examples)])  # best-first prefixes of the input
    assert layout != PromptLayout.QUESTION_ONLY or not packed.has_context  # question-only holds no evidence
    for pair in packed.pair_examples:  # examples are never cut
        assert pair.record.answer in packed.prompt_text and pair.record.question in packed.prompt_text  # whole
    for passage in packed.passages[:-1]:  # only the last passage may be shorter than its record
        assert passage.record.text in packed.prompt_text  # whole


def test_second_passage_is_trimmed_when_the_budget_runs_out(evidence_factory, gpt2_tokenizer):  # the first passage is whole, the last one is cut at a sentence end
    first = evidence_factory["passage"]("bp-1", evidence_factory["text"](random.Random(2), 8), 0, 0)  # about 150 tokens
    second = evidence_factory["passage"]("bp-2", evidence_factory["text"](random.Random(3), 12), 0, 50_000)  # does not fit whole
    budget = PromptBudget(CONTEXT_LENGTH, 120, 260, 8)  # room for the first and a piece of the second
    packed = pack_prompt("Who carried the lantern?", evidence_factory["evidence"]([first, second], []), PromptLayout.PASSAGES_BEFORE_TEMPLATE, budget, gpt2_tokenizer)  # pack with a budget that only fits part of the second passage
    assert [p.record.passage_id for p in packed.passages] == ["bp-1", "bp-2"] and first.record.text in packed.prompt_text  # first whole
    assert second.record.text not in packed.prompt_text  # second is shorter than its record
    trimmed_block = packed.prompt_text.split("\n\n")[0]  # the worst passage is rendered first
    assert second.record.text.startswith(trimmed_block) and trimmed_block != second.record.text and trimmed_block.endswith(".")  # a sentence-aligned prefix
    assert count_tokens(trimmed_block, gpt2_tokenizer) >= MINIMUM_TRIMMED_TOKENS  # worth showing


def test_a_remainder_under_forty_tokens_is_dropped(evidence_factory, gpt2_tokenizer):  # no tiny fragments
    first = evidence_factory["passage"]("bp-1", evidence_factory["text"](random.Random(2), 8), 0, 0)  # fills most of the budget
    second = evidence_factory["passage"]("bp-2", evidence_factory["text"](random.Random(3), 12), 0, 50_000)  # would get a tiny remainder
    first_tokens = count_tokens(first.record.text, gpt2_tokenizer)  # size of the first passage
    budget = PromptBudget(CONTEXT_LENGTH, 120, first_tokens + 2 + 20, 8)  # only 20 tokens left for the second
    packed = pack_prompt("Who carried the lantern?", evidence_factory["evidence"]([first, second], []), PromptLayout.PASSAGES_BEFORE_TEMPLATE, budget, gpt2_tokenizer)  # pack with almost no room for the second passage
    assert [p.record.passage_id for p in packed.passages] == ["bp-1"]  # the fragment was left out


def test_more_than_half_overlap_is_skipped_but_adjacent_chunks_are_kept(evidence_factory, gpt2_tokenizer):  # same part, character spans compared
    text = evidence_factory["text"](random.Random(4), 6)  # one passage text
    base = evidence_factory["passage"]("bp-1", text, 0, 1000)  # spans 1000 .. 1000 + n
    heavy = evidence_factory["passage"]("bp-2", text, 0, 1000 + len(text) // 4)  # three quarters of it overlap
    adjacent = evidence_factory["passage"]("bp-3", text, 0, 1000 + len(text) * 4 // 5)  # about a fifth overlaps
    other_part = evidence_factory["passage"]("bp-4", text, 1, 1000)  # same offsets, another part: not an overlap
    packed = pack_prompt("Q?", evidence_factory["evidence"]([base, heavy, adjacent, other_part], []), PromptLayout.PASSAGES_BEFORE_TEMPLATE,
                         PromptBudget(CONTEXT_LENGTH, 120, 700, 8), gpt2_tokenizer)  # roomy budget
    assert [p.record.passage_id for p in packed.passages] == ["bp-1", "bp-3", "bp-4"]  # the heavy copy was skipped


def test_layouts_ignore_item_kinds_they_do_not_use(evidence_factory, gpt2_tokenizer):  # and the packer does not list them as sources
    passage = evidence_factory["passage"]("bp-1", evidence_factory["text"](random.Random(5), 3), 0, 0)  # one passage
    pair = evidence_factory["pair"]("qa-1", "Who carried the lantern?", "Brannoc carried it.")  # one example
    evidence = evidence_factory["evidence"]([passage], [pair])  # both kinds
    budget = PromptBudget(CONTEXT_LENGTH, 120, 600, 8)  # roomy
    only_passages = pack_prompt("Q?", evidence, PromptLayout.PASSAGES_BEFORE_TEMPLATE, budget, gpt2_tokenizer)  # passage layout
    only_pairs = pack_prompt("Q?", evidence, PromptLayout.PAIR_EXAMPLES_BEFORE_QUESTION, budget, gpt2_tokenizer)  # pair layout
    both = pack_prompt("Q?", evidence, PromptLayout.PASSAGES_AND_PAIR_EXAMPLES, budget, gpt2_tokenizer)  # combined layout
    nothing = pack_prompt("Q?", evidence, PromptLayout.QUESTION_ONLY, budget, gpt2_tokenizer)  # question only
    assert (len(only_passages.passages), len(only_passages.pair_examples)) == (1, 0)  # passages only
    assert (len(only_pairs.passages), len(only_pairs.pair_examples)) == (0, 1)  # examples only
    assert (len(both.passages), len(both.pair_examples)) == (1, 1) and not nothing.has_context  # both, and none


def test_zero_budget_gives_the_plain_prompt(evidence_factory, gpt2_tokenizer):  # no room means no evidence, never an error
    passage = evidence_factory["passage"]("bp-1", evidence_factory["text"](random.Random(6), 3), 0, 0)  # one passage
    packed = pack_prompt("Q?", evidence_factory["evidence"]([passage], []), PromptLayout.PASSAGES_BEFORE_TEMPLATE, PromptBudget(CONTEXT_LENGTH, 120, 0, 8), gpt2_tokenizer)  # pack with a zero budget
    assert packed.prompt_text == render_question_block("Q?") and not packed.has_context  # today's prompt
