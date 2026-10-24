# The `answers` stage: the selection grid on fact-development and G1 development questions (four layouts at budget 600, then the best layout
# at 300 and 450), then the baseline and the chosen configuration ONCE on G2, G4 and G1 test. Writes results/answers.json.
import dataclasses  # result dictionaries
from typing import Any, Mapping, Sequence
from retrieval_contracts.bundle_manifest import PAIR_RECORDS_FILE_NAME, PASSAGE_RECORDS_FILE_NAME  # bundle file names
from retrieval_contracts.bundle_records import PassageRecord, QuestionAnswerRecord, read_json_lines  # stored records
from retrieval_evaluation.evaluation_settings import EvaluationSettings  # numbers
from retrieval_evaluation.evaluation_stages.calibration_stage import build_calibrated_settings  # chosen retrieval with calibrated values
from retrieval_evaluation.evaluation_stages.evaluation_runtime import EvaluationRuntime  # service objects
from retrieval_evaluation.evaluation_stages.retrieval_stage import build_chosen_search_settings  # chosen mode and reranker
from retrieval_evaluation.experiments.answer_run_planning import (  # plan, clock and items
    AnswerConfiguration, GenerationClock, build_answer_items, build_configuration_name, count_generations, plan_held_out_items,
)
from retrieval_evaluation.experiments.answer_runs import AnswerRunTools, run_answer_items  # the runner
from retrieval_evaluation.experiments.evaluation_report_writing import write_json_results  # result files
from retrieval_evaluation.experiments.settings_recommendation import ConfigurationSelectionScore, choose_layout_and_budget  # selection rule
from retrieval_evaluation.golden_sets.golden_question_records import GoldenSets  # question sets
from retrieval_evaluation.golden_sets.regression_questions import IRON_THRONE_REJECTED_TERMS, is_iron_throne_question  # A4 words
from retrieval_evaluation.metrics.answer_run_scoring import AnswerRecord, score_fact_records, score_held_out_records  # scoring
from retrieval_evaluation.metrics.copyright_checks import build_passage_run_hashes, find_copyright_problems  # A16
from retrieval_evaluation.metrics.fact_rule_loading import FactRuleBook  # word rules

ANSWERS_FILE_NAME = "answers.json"  # inside the results folder
BASELINE_LAYOUT = "question_only"  # today's prompt: the no-retrieval arm
RETRIEVAL_LAYOUTS = ("passages_before_template", "pair_examples_before_question", "passages_and_pair_examples")  # layouts that use evidence
PASSAGE_ONLY_LAYOUTS = ("passages_before_template",)  # the only evidence layout left when no pair threshold reached the precision target


def make_configurations(layouts: Mapping[str, Any], layout_names: Sequence[str], budgets: Sequence[int]) -> list[AnswerConfiguration]:  # layout x budget
    return [AnswerConfiguration(build_configuration_name(name, budget), layouts[name], budget) for name in layout_names for budget in budgets]  # grid


def build_lookup_helpers(bundle_folder: Any) -> tuple[Any, Any]:  # (find supporting texts by id, find A16 problems against ALL passages)
    passages = read_json_lines(bundle_folder / PASSAGE_RECORDS_FILE_NAME, PassageRecord)  # every passage of the bundle
    pairs = read_json_lines(bundle_folder / PAIR_RECORDS_FILE_NAME, QuestionAnswerRecord)  # every pair
    texts_by_id = {passage.passage_id: passage.text for passage in passages}  # passage id -> text
    texts_by_id.update({pair.pair_id: pair.question + " " + pair.answer for pair in pairs})  # pair id -> question and answer
    run_hashes = build_passage_run_hashes(passage.text for passage in passages)  # 26-word runs of every passage
    find_texts = lambda passage_ids, pair_ids: [texts_by_id[item_id] for item_id in (*passage_ids, *pair_ids)]  # texts of the packed items
    return find_texts, lambda completion, sources: find_copyright_problems(completion, sources, run_hashes)  # A16 closure


def score_selection(records: Sequence[AnswerRecord], configuration: AnswerConfiguration, golden: GoldenSets, rule_book: FactRuleBook) -> ConfigurationSelectionScore:  # scores one configuration on the selection sets
    own = [record for record in records if record.configuration_name == configuration.name]  # this configuration's answers
    rules = {question.question_id: question.rule for question in golden.fact_development}  # rules by id
    references = {question.question_id: (question.question, question.gold_answer) for question in golden.held_out_development}  # (question, gold answer) by id
    facts = score_fact_records([record for record in own if record.set_name == "fact_development"], rule_book, rules)  # majority score
    held_out = score_held_out_records([record for record in own if record.set_name == "held_out_development"], references)  # recall and prompt size
    return ConfigurationSelectionScore(configuration.name, str(configuration.layout), configuration.context_token_budget, facts.majority_right,
                                       held_out.key_term_recall, held_out.mean_prompt_tokens)


def score_final_arm(records: Sequence[AnswerRecord], golden: GoldenSets, rule_book: FactRuleBook) -> dict[str, Any]:  # everything the ship gate reads for one arm
    of_set = lambda name: [record for record in records if record.set_name == name]  # records of one question set
    facts = score_fact_records(of_set("fact"), rule_book, {question.question_id: question.rule for question in golden.fact})  # G2
    regression = score_fact_records(of_set("regression"), rule_book, {question.question_id: question.rule for question in golden.regression})  # G4
    iron_throne_ids = {question.question_id for question in golden.regression if is_iron_throne_question(question)}  # A4 questions
    names_mountain = any(rule_book.term_occurs(term, record.completion) for record in of_set("regression") if record.question_id in iron_throne_ids
                         for term in IRON_THRONE_REJECTED_TERMS)  # any sample names the wrong man
    held_out = score_held_out_records(of_set("held_out_test"), {question.question_id: (question.question, question.gold_answer) for question in golden.held_out_test})  # G1 test
    return {"facts": dataclasses.asdict(facts), "known_ten": sum(question.is_known_ten and question.question_id in facts.right_question_ids for question in golden.fact),
            "facts_without_pairs": sum(not question.is_word_for_word_in_pairs and question.question_id in facts.right_question_ids for question in golden.fact),
            "regression": dataclasses.asdict(regression), "iron_throne_names_mountain": names_mountain, "held_out": dataclasses.asdict(held_out),
            "copyright_problem_count": sum(len(record.copyright_problems) for record in records)}


def run_answers_stage(golden: GoldenSets, ablation: dict[str, Any], calibration: dict[str, Any], runtime: EvaluationRuntime, rule_book: FactRuleBook,  # selection grid, then the final arms
                      settings: EvaluationSettings) -> dict[str, Any]:
    run, bundle_name, seed = settings.answers, ablation["decisions"]["chunk_size_name"], settings.golden.pair_sample_seed  # shortcuts
    calibrated = build_calibrated_settings(build_chosen_search_settings(runtime.shipped_search_settings, ablation["decisions"]), calibration)  # as shipped
    find_texts, find_problems = build_lookup_helpers(runtime.bundle_folders[bundle_name])  # text lookup and A16 check
    tools = AnswerRunTools(runtime.build_retriever(bundle_name, calibrated), runtime.answer_service, runtime.build_answer_request,
                           runtime.make_random_generator, find_texts, find_problems)  # one tool bundle for the whole stage
    run_items = lambda items, configurations, clock: run_answer_items(items, configurations, tools, clock, run.seed_multiplier, runtime.report_progress)
    development_items = build_answer_items("held_out_development", [(question.question_id, question.question) for question in golden.held_out_development], run.held_out_sample_count)  # G1 development items
    fact_development_items = build_answer_items("fact_development", [(question.question_id, question.rule.question) for question in golden.fact_development], run.fact_sample_count)  # fact-development items
    pairs_available = calibration["pair_threshold"]["threshold"] is not None  # without a threshold no pair example ever reaches a prompt
    evidence_layouts = RETRIEVAL_LAYOUTS if pairs_available else PASSAGE_ONLY_LAYOUTS  # the others would repeat the baseline or the passage prompt
    first_grid = make_configurations(runtime.layouts, (BASELINE_LAYOUT,) + evidence_layouts, (run.selection_layout_budget,))  # layouts at 600
    grid_size = len(first_grid) + len(run.reduced_budgets)  # configurations after the second step
    clock = GenerationClock()  # times the first generations
    records = run_items(fact_development_items, first_grid, clock)  # fact development first: these are timed
    plan = plan_held_out_items(count_generations(fact_development_items, grid_size), development_items, grid_size, clock, run, seed)  # cap check
    records += run_items(plan.kept_items, first_grid, clock)  # G1 development under the first grid
    best_layout = choose_layout_and_budget([score_selection(records, configuration, golden, rule_book) for configuration in first_grid if str(configuration.layout) != BASELINE_LAYOUT])  # best layout
    reduced_grid = make_configurations(runtime.layouts, (best_layout.layout_name,), run.reduced_budgets)  # same layout, smaller budgets
    records += run_items(fact_development_items + list(plan.kept_items), reduced_grid, clock)  # second step
    grid = first_grid + reduced_grid  # every configuration
    selection = [score_selection(records, configuration, golden, rule_book) for configuration in grid]  # scores of all six
    chosen = choose_layout_and_budget(selection)  # best fact-development majority (rule 3)
    baseline = first_grid[0]  # question_only
    final_arms = [baseline] + [configuration for configuration in grid if configuration.name == chosen.name and configuration.name != baseline.name]  # chosen only once
    final_items = (build_answer_items("fact", [(question.question_id, question.rule.question) for question in golden.fact], run.fact_sample_count)  # G2
                   + build_answer_items("regression", [(question.question_id, question.rule.question) for question in golden.regression], run.fact_sample_count)  # G4
                   + build_answer_items("held_out_test", [(question.question_id, question.question) for question in golden.held_out_test], run.held_out_sample_count))  # G1 test
    final_records = run_items(final_items, final_arms, clock)  # reported once
    arm_scores = {arm.name: score_final_arm([record for record in final_records if record.configuration_name == arm.name], golden, rule_book) for arm in final_arms}  # per arm
    result = {"selection": [dataclasses.asdict(score) for score in selection], "chosen": chosen.name, "baseline": baseline.name,
              "plan": {"reason": plan.reason + ("" if pairs_available else "; pair-example layouts skipped because no pair threshold reached the precision target"), "was_reduced": plan.was_reduced, "projected_hours": plan.projected_hours, "kept_question_count": len(plan.kept_items)},
              "final": {"baseline": arm_scores[baseline.name], "chosen": arm_scores[chosen.name]}, "generation_count": len(records) + len(final_records),
              "records": [dataclasses.asdict(record) for record in records + final_records]}
    write_json_results(settings.locations.results_folder / ANSWERS_FILE_NAME, result)  # record
    return result  # for the report stage
