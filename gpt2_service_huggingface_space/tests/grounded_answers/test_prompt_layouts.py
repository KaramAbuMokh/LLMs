# Tests that the prompt text is byte-exact: question-only equals the prompt the service used before retrieval, the other layouts order items best-last
from gpt2_demo_service.grounded_answers.answer_settings import PromptLayout  # layout names
from gpt2_demo_service.grounded_answers.prompt_layouts import render_pair_example, render_prompt, render_question_block  # code under test

TODAYS_PROMPT = (  # the literal text of the old service's template, "qusetion" typo included
    "Answer the following question with an appropriate response. Write a response that appropriately completes the qusetion."
    "\n\n### question:\nWho is Jon Snow?\n\n### Response:\n"
)


def test_question_only_is_todays_prompt_byte_for_byte():  # nothing may change for the trained template
    assert render_question_block("Who is Jon Snow?") == TODAYS_PROMPT  # same bytes
    assert render_question_block("  Who is Jon Snow? \n") == TODAYS_PROMPT  # the old code stripped the question
    assert render_prompt(PromptLayout.QUESTION_ONLY, "Who is Jon Snow?", ["a passage"], [("q", "a")]) == TODAYS_PROMPT  # items ignored


def test_pair_example_is_the_question_block_plus_the_answer():  # in-format example without an end token
    assert render_pair_example("Q?", "An answer.") == render_question_block("Q?") + "An answer."  # block then answer


def test_passage_layout_puts_the_best_passage_next_to_the_question():  # inputs are best first, rendering is best last
    prompt = render_prompt(PromptLayout.PASSAGES_BEFORE_TEMPLATE, "Q?", ["best", "second", "third"], [("x", "y")])  # examples are ignored here
    assert prompt == "third\n\nsecond\n\nbest\n\n" + render_question_block("Q?")  # worst first


def test_pair_layout_puts_the_best_example_next_to_the_question():  # same rule for examples
    prompt = render_prompt(PromptLayout.PAIR_EXAMPLES_BEFORE_QUESTION, "Q?", ["ignored"], [("q1", "a1"), ("q2", "a2")])  # passages are ignored here
    assert prompt == render_pair_example("q2", "a2") + "\n\n" + render_pair_example("q1", "a1") + "\n\n" + render_question_block("Q?")  # worst example first


def test_combined_layout_puts_passages_then_examples_then_the_question():  # the in-format example sits nearest
    prompt = render_prompt(PromptLayout.PASSAGES_AND_PAIR_EXAMPLES, "Q?", ["p1", "p2"], [("q1", "a1")])  # both kinds
    assert prompt == "p2\n\np1\n\n" + render_pair_example("q1", "a1") + "\n\n" + render_question_block("Q?")  # passages, example, question


def test_layouts_without_items_fall_back_to_the_question_block():  # empty evidence renders today's prompt
    for layout in PromptLayout:  # all four
        assert render_prompt(layout, "Who is Jon Snow?", [], []) == TODAYS_PROMPT  # unchanged
