# The prompt text: the EXACT trained question template, plus the four ways retrieved text can sit in front of it
from typing import Sequence  # lists of retrieved texts
from gpt2_demo_service.grounded_answers.answer_settings import PromptLayout  # layout names

TRAINING_PROMPT_HEADER = (  # the text every training example started with
    "Answer the following question with an appropriate response. "
    "Write a response that appropriately completes the qusetion."  # "qusetion" is spelled like this in the training data
)
BLOCK_SEPARATOR = "\n\n"  # blank line between retrieved items and before the question block
PASSAGE_LAYOUTS = frozenset({PromptLayout.PASSAGES_BEFORE_TEMPLATE, PromptLayout.PASSAGES_AND_PAIR_EXAMPLES})  # layouts that show passages
PAIR_LAYOUTS = frozenset({PromptLayout.PAIR_EXAMPLES_BEFORE_QUESTION, PromptLayout.PASSAGES_AND_PAIR_EXAMPLES})  # layouts that show examples


def render_question_block(question: str) -> str:  # byte-identical to the prompt the service used before retrieval existed
    return TRAINING_PROMPT_HEADER + "\n\n### question:\n" + question.strip() + "\n\n### Response:\n"  # header, question, response marker


def render_pair_example(question: str, answer: str) -> str:  # one solved example in the trained format (no end token)
    return render_question_block(question) + answer  # the answer follows the response marker


def render_prompt(layout: PromptLayout, question: str, passage_texts: Sequence[str], pair_examples: Sequence[tuple[str, str]]) -> str:
    # inputs are best first; the best item is rendered LAST, next to the question; a layout ignores item kinds it does not use
    blocks = []  # rendered items, furthest from the question first
    if layout in PASSAGE_LAYOUTS:  # passages are shown
        blocks.extend(reversed(passage_texts))  # best passage ends up nearest
    if layout in PAIR_LAYOUTS:  # examples are shown
        blocks.extend(render_pair_example(example_question, example_answer) for example_question, example_answer in reversed(pair_examples))  # best example nearest
    blocks.append(render_question_block(question))  # the question block always comes last
    return BLOCK_SEPARATOR.join(blocks)  # items separated by blank lines
