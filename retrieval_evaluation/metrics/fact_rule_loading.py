# Loads the 30 fact questions and their word rules from the Kaggle source, so every answer is judged by the SAME is_right that produced 10/30
import importlib.util  # loads a Python file by path
import sys  # temporary module registration: fact_questions.py imports run_settings by name
from dataclasses import dataclass  # frozen rule records
from pathlib import Path  # source file paths
from types import ModuleType  # loaded module type
from typing import Callable  # stored functions

FACT_QUESTIONS_MODULE_NAME = "fact_questions"  # the name the source file expects
RUN_SETTINGS_MODULE_NAME = "run_settings"  # the module fact_questions.py imports at its top
KNOWN_TEN_COUNT = 10  # the first ten fact questions are the known ten


@dataclass(frozen=True)
class FactRule:  # one question with the word rules that mark an answer right
    question: str  # the question text
    accepted_groups: tuple[tuple[str, ...], ...]  # every group needs at least one matching term
    rejected_terms: tuple[str, ...]  # none of these may appear


@dataclass(frozen=True)
class FactRuleBook:  # the loaded rules plus the source module's own matching functions
    rules: tuple[FactRule, ...]  # the 30 rules in source order
    judge: Callable[[str, list, list], bool]  # the source's is_right(answer, accepted_groups, rejected_terms)
    term_matcher: Callable[[str], object]  # the source's term_pattern(term): a compiled whole-word regex

    def is_answer_right(self, rule: FactRule, answer: str) -> bool:  # True when the answer satisfies the rule (same function as the Kaggle scoring)
        return self.judge(answer, [list(group) for group in rule.accepted_groups], list(rule.rejected_terms))  # delegate

    def term_occurs(self, term: str, text: str) -> bool:  # True when the rule term matches the text the way is_right matches it
        return self.term_matcher(term).search(text.lower().replace("’", "'")) is not None  # same folding as is_right


def load_module_from_path(module_name: str, path: Path) -> ModuleType:  # imports one source file under a given module name
    spec = importlib.util.spec_from_file_location(module_name, path)  # where and how to load it
    module = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[module_name] = module  # registered while it runs (dataclasses and sibling imports look it up)
    spec.loader.exec_module(module)  # runs the file
    return module  # ready to use


def build_fact_rule(question: str, accepted_groups: list, rejected_terms: list) -> FactRule:  # one frozen rule from the source's lists
    return FactRule(question, tuple(tuple(group) for group in accepted_groups), tuple(rejected_terms))  # immutable copy


def load_fact_rule_book(fact_rules_path: Path, run_settings_path: Path) -> FactRuleBook:  # the 30 rules of the Kaggle source
    try:  # modules are registered under their plain names only while loading
        load_module_from_path(RUN_SETTINGS_MODULE_NAME, run_settings_path)  # fact_questions imports FIXED_QUESTIONS from it
        fact_module = load_module_from_path(FACT_QUESTIONS_MODULE_NAME, fact_rules_path)  # the questions and is_right
    finally:  # leave no trace in the import system
        sys.modules.pop(RUN_SETTINGS_MODULE_NAME, None)  # forget run_settings
        sys.modules.pop(FACT_QUESTIONS_MODULE_NAME, None)  # forget fact_questions
    rules = tuple(build_fact_rule(*entry) for entry in fact_module.FACT_QUESTIONS)  # (question, groups, rejected) per entry
    return FactRuleBook(rules, fact_module.is_right, fact_module.term_pattern)  # rules plus the source's own matchers
