# Plan arithmetic: items, seeded subsets, the generation clock, and halving to fit the hour and generation caps (DESIGN.md §11.5)
from retrieval_evaluation.evaluation_settings import AnswerRunSettings  # caps
from retrieval_evaluation.experiments.answer_run_planning import (  # under test
    GenerationClock, build_answer_items, build_configuration_name, count_generations, plan_held_out_items, select_seeded_subset,
)

QUESTIONS = [("q%03d" % number, "question %d" % number) for number in range(300)]  # 300 held-out questions


def test_items_carry_their_position_and_count_their_generations():  # position = part of the seed
    items = build_answer_items("held_out_development", QUESTIONS[:5], 2)  # five questions, two samples
    assert [item.question_index for item in items] == [0, 1, 2, 3, 4] and items[0].set_name == "held_out_development"  # positions
    assert count_generations(items, 6) == 5 * 2 * 6  # samples times configurations
    assert build_configuration_name("question_only", 600) == "question_only_600"  # readable name


def test_seeded_subset_is_reproducible_and_keeps_the_original_order():  # same seed, same questions
    items = build_answer_items("s", QUESTIONS, 1)  # 300 items
    first, second = select_seeded_subset(items, 40, seed=7), select_seeded_subset(items, 40, seed=7)  # twice
    assert first == second and len(first) == 40  # reproducible
    assert [item.question_index for item in first] == sorted(item.question_index for item in first)  # original order
    assert first != select_seeded_subset(items, 40, seed=8)  # another seed, another subset


def test_clock_projection_uses_only_the_probe_generations():  # the first 30 are timed
    clock = GenerationClock()  # empty
    assert clock.mean_seconds(30) is None  # nothing yet
    for seconds in [2.0] * 30 + [100.0] * 10:  # 30 probe generations, then slow ones
        clock.record(seconds)  # record
    assert clock.mean_seconds(30) == 2.0  # later ones do not count


def test_plan_keeps_everything_when_it_fits():  # no reduction
    clock = GenerationClock()  # timing
    clock.record(5.0)  # 5 seconds each
    items = build_answer_items("h", QUESTIONS[:100], 1)  # 100 items
    decision = plan_held_out_items(100, items, 6, clock, AnswerRunSettings(), seed=1)  # 100 + 600 generations = 3.5 hours at 5 s
    assert not decision.was_reduced and len(decision.kept_items) == 100 and decision.projected_hours < 6.0  # fits


def test_plan_halves_the_held_out_subset_above_six_hours_and_below_the_generation_cap():  # both caps
    clock = GenerationClock()  # timing
    clock.record(100.0)  # 100 seconds each: far too slow
    items = build_answer_items("h", QUESTIONS, 1)  # 300 items
    decision = plan_held_out_items(100, items, 6, clock, AnswerRunSettings(), seed=1)  # 1,900 generations is 52 hours
    assert decision.was_reduced and decision.projected_hours <= 6.0 and len(decision.kept_items) < 300  # halved until it fits
    assert "cut from 300" in decision.reason  # the report says so
    uncapped_clock = GenerationClock()  # nothing timed
    huge = plan_held_out_items(0, build_answer_items("h", QUESTIONS * 3, 1), 6, uncapped_clock, AnswerRunSettings(), seed=1)  # 5,400 generations
    assert 6 * len(huge.kept_items) <= 3000 and huge.projected_hours is None  # hard cap without any timing
