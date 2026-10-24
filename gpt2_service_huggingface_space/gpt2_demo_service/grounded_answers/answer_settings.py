# Every number the answer path consumes, declared WITHOUT defaults (the shipped values live in service_settings.py)
from dataclasses import dataclass  # frozen settings record
from enum import StrEnum  # layout names as readable strings


class PromptLayout(StrEnum):  # where retrieved text sits around the trained question template (DESIGN.md 7.3)
    QUESTION_ONLY = "question_only"  # today's prompt: the baseline and the fallback
    PASSAGES_BEFORE_TEMPLATE = "passages_before_template"  # raw book passages, then the question block
    PAIR_EXAMPLES_BEFORE_QUESTION = "pair_examples_before_question"  # solved examples in the trained format, then the question block
    PASSAGES_AND_PAIR_EXAMPLES = "passages_and_pair_examples"  # passages, then examples, then the question block


@dataclass(frozen=True)
class GroundedAnswerSettings:  # how prompts are packed and answers are cited
    layout: PromptLayout  # where retrieved text sits
    context_token_budget: int  # most tokens of retrieved text in a prompt
    safety_margin_tokens: int  # tokens kept free beyond the answer reserve
    answer_token_floor: int  # shortest answer reserve (answers end early on their own)
    answer_token_ceiling: int  # longest answer reserve; reserve = min(max(request tokens, floor), ceiling)
    book_snippet_enabled: bool  # the first passage source may carry a snippet of at most 25 words
