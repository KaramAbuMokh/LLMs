# What an answer run will generate and whether it fits the time and count caps (DESIGN.md §11.5): items, configurations, the generation
# clock that projects the total hours after the first timed generations, and the seeded halving of the G1 development subset.
import random  # seeded subset
from dataclasses import dataclass  # frozen records
from typing import Any, Sequence  # inputs
from retrieval_evaluation.evaluation_settings import AnswerRunSettings  # caps and sample counts


@dataclass(frozen=True)
class AnswerItem:  # one question to answer in a run
    question_id: str  # id in its question set
    set_name: str  # the question set, e.g. "fact_development"
    question: str  # the question text
    question_index: int  # position inside its set: part of every sample's seed
    sample_count: int  # samples per configuration


@dataclass(frozen=True)
class AnswerConfiguration:  # one prompt layout at one context budget
    name: str  # e.g. "passages_and_pair_examples_600"
    layout: Any  # a PromptLayout of the service (a plain string in tests)
    context_token_budget: int  # retrieved tokens allowed in the prompt


@dataclass(frozen=True)
class PlanDecision:  # which G1 development questions to run and why
    kept_items: tuple[AnswerItem, ...]  # the items that will run
    was_reduced: bool  # the subset was cut below the full set
    projected_hours: float | None  # projection for the kept plan (None when nothing was timed)
    reason: str  # one sentence for the report


class GenerationClock:  # collects the seconds of every generation so far
    def __init__(self) -> None:  # starts empty
        self.seconds: list[float] = []  # one entry per generation

    def record(self, seconds: float) -> None:  # called after each generation
        self.seconds.append(seconds)  # remember it

    def mean_seconds(self, probe_count: int) -> float | None:  # mean of the first probe_count generations (None before any)
        probe = self.seconds[:probe_count]  # the timing probe
        return sum(probe) / len(probe) if probe else None  # mean seconds per generation


def build_configuration_name(layout_name: str, context_token_budget: int) -> str:  # "<layout>_<budget>"
    return "%s_%d" % (layout_name, context_token_budget)  # readable and unique


def build_answer_items(set_name: str, questions: Sequence[tuple[str, str]], sample_count: int) -> list[AnswerItem]:  # (id, text) pairs to items
    return [AnswerItem(question_id, set_name, text, index, sample_count) for index, (question_id, text) in enumerate(questions)]  # set order = seed order


def count_generations(items: Sequence[AnswerItem], configuration_count: int) -> int:  # generations the items need
    return sum(item.sample_count for item in items) * configuration_count  # every sample under every configuration


def select_seeded_subset(items: Sequence[AnswerItem], keep_count: int, seed: int) -> list[AnswerItem]:  # reproducible subset in the original order
    chosen_positions = sorted(random.Random(seed).sample(range(len(items)), min(keep_count, len(items))))  # seeded positions
    return [items[position] for position in chosen_positions]  # original order kept


def plan_held_out_items(fixed_generations: int, held_out_items: Sequence[AnswerItem], configuration_count: int, clock: GenerationClock,
                        settings: AnswerRunSettings, seed: int) -> PlanDecision:
    """Halves the held-out subset until the projected total fits the hours limit and the generation cap; fixed_generations already ran or will."""
    kept = list(held_out_items)  # start with all of them
    mean_seconds = clock.mean_seconds(settings.timing_probe_count)  # None before any timing
    while True:  # halve until it fits
        total = fixed_generations + count_generations(kept, configuration_count)  # generations of the whole run
        hours = None if mean_seconds is None else mean_seconds * total / 3600.0  # projection
        too_many = total > settings.generation_hard_cap  # the hard cap
        too_slow = hours is not None and hours > settings.projection_limit_hours  # the time limit
        if not (too_many or too_slow) or len(kept) <= 1:  # fits (or nothing left to cut)
            break  # done
        kept = select_seeded_subset(kept, len(kept) // 2, seed)  # halve, seeded
    was_reduced = len(kept) < len(held_out_items)  # something was cut
    reason = ("held-out subset cut from %d to %d questions to fit the %.0f hour and %d generation caps" % (
        len(held_out_items), len(kept), settings.projection_limit_hours, settings.generation_hard_cap)) if was_reduced else "full held-out set fits the caps"
    return PlanDecision(tuple(kept), was_reduced, hours, reason)  # the decision
