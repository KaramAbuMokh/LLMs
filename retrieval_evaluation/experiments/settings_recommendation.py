# The decision rules of DESIGN.md §11.6 as pure functions: search mode, reranker, chunk size, prompt layout and budget, the ship gate and the
# rule that says whether a grounding fine-tune is needed. Inputs are measured numbers; outputs are choices with a one-line reason.
from dataclasses import dataclass  # frozen records
from typing import Any, Mapping, Sequence  # inputs
from retrieval_evaluation.evaluation_settings import RetrievalMeasureSettings, ShipGateSettings  # rule numbers

HYBRID_NAME = "hybrid"  # the default search mode (D7)
SINGLE_SEARCH_NAMES = ("dense_only", "lexical_only")  # the searches that may replace it


@dataclass(frozen=True)  # immutable record
class ConfigurationSelectionScore:  # how one layout and budget did on the selection sets
    name: str  # configuration name
    layout_name: str  # prompt layout value
    context_token_budget: int  # retrieved tokens allowed
    fact_development_majority_right: int  # fact-development questions right by majority
    held_out_key_term_recall: float  # mean key-term recall on G1 development
    mean_prompt_tokens: float  # mean packed prompt length


@dataclass(frozen=True)  # immutable record
class ShipMeasurements:  # every number the ship gate reads, measured once on the chosen configuration
    facts_right: int  # G2 questions right by majority (of 30)
    facts_without_pairs_right: int  # the same without the 3 word-for-word pair questions (of 27)
    known_ten_right: int  # first ten of G2 (of 10)
    turned_wrong_count: int  # questions the baseline got right that the chosen configuration gets wrong
    iron_throne_names_mountain: bool  # a G4 Iron Throne answer named Gregor, Clegane or the Mountain
    test_hit_at_5: float  # passage hit-at-5 on G1 test
    unanswerable_evidence_share: float  # G3 questions that still get evidence
    development_evidence_share: float  # G1 development questions that get evidence
    fact_development_evidence_share: float  # fact-development questions that get evidence
    pair_precision: float | None  # held-back half precision at the chosen pair threshold
    copyright_problem_count: int  # A16 findings over every evaluation answer
    chosen_key_term_recall: float  # G1 test key-term recall of the chosen configuration
    baseline_key_term_recall: float  # G1 test key-term recall of question_only


@dataclass(frozen=True)  # immutable record
class GateResult:  # one line of the ship gate
    gate: str  # "A1", "A7b", ...
    status: str  # "pass", "fail" or "live_check" (measured on the Space after deploy)
    detail: str  # the number and the threshold


def choose_search_mode(hit_at_5_by_name: Mapping[str, float], measure: RetrievalMeasureSettings) -> str:  # hybrid unless a single search beats it enough
    best_name = max(SINGLE_SEARCH_NAMES, key=lambda name: hit_at_5_by_name[name])  # the better single search
    beats_hybrid = hit_at_5_by_name[best_name] - hit_at_5_by_name[HYBRID_NAME] >= measure.minimum_search_mode_gain - 1e-9  # by the required margin
    return best_name if beats_hybrid else HYBRID_NAME  # D7


def decide_reranker(hybrid_hit_at_3: float, rerank_hit_at_3: float, rerank_p95_milliseconds: float, measure: RetrievalMeasureSettings) -> tuple[bool, str]:  # rule 2: reranker on or off
    gain = rerank_hit_at_3 - hybrid_hit_at_3  # hit-at-3 points gained
    estimated_milliseconds = rerank_p95_milliseconds * measure.space_slowdown_factor  # Space estimate (A9, reranker on)
    enough_gain = gain >= measure.minimum_reranker_gain - 1e-9  # +3 points
    fast_enough = estimated_milliseconds <= measure.reranker_on_limit_milliseconds  # A9 holds
    return enough_gain and fast_enough, "hit-at-3 gain %.3f (needs %.2f), estimated Space p95 %.0f ms (limit %.0f)" % (  # one sentence
        gain, measure.minimum_reranker_gain, estimated_milliseconds, measure.reranker_on_limit_milliseconds)  # values filled into the sentence above


def choose_chunk_size_name(packed_hit_by_bundle: Mapping[str, float], shipped_name: str, measure: RetrievalMeasureSettings) -> str:  # keeps the shipped size unless clearly beaten
    best_name = max(packed_hit_by_bundle, key=lambda name: packed_hit_by_bundle[name])  # best packed-budget hit
    gain = packed_hit_by_bundle[best_name] - packed_hit_by_bundle[shipped_name]  # over the shipped chunk size
    return best_name if gain >= measure.minimum_chunk_size_gain - 1e-9 else shipped_name  # keep the shipped size unless clearly beaten


def choose_layout_and_budget(scores: Sequence[ConfigurationSelectionScore]) -> ConfigurationSelectionScore:  # picks the best configuration of the selection grid
    """Best fact-development majority score; ties by G1 development key-term recall, then by fewer prompt tokens."""
    return max(scores, key=lambda score: (score.fact_development_majority_right, score.held_out_key_term_recall, -score.mean_prompt_tokens))  # best by the three-part key


def decide_raft(facts_right: int, known_ten_right: int, recall_gain: float, gate: ShipGateSettings) -> tuple[bool, list[str]]:  # rule 5: is a grounding fine-tune needed
    reasons = []  # which triggers fired
    if facts_right < gate.raft_facts_minimum:  # under 20 of 30
        reasons.append("facts %d of 30 is under %d" % (facts_right, gate.raft_facts_minimum))  # trigger 1
    if known_ten_right < gate.raft_known_ten_minimum:  # under 7 of 10
        reasons.append("known ten %d of 10 is under %d" % (known_ten_right, gate.raft_known_ten_minimum))  # trigger 2
    if recall_gain < gate.raft_recall_gain_minimum:  # under 10 points
        reasons.append("key-term recall gain %.3f is under %.2f" % (recall_gain, gate.raft_recall_gain_minimum))  # trigger 3
    return bool(reasons), reasons  # needed when any fired


def check_gate(gate_name: str, passed: bool, detail: str) -> GateResult:  # one result line
    return GateResult(gate_name, "pass" if passed else "fail", detail)  # status from the boolean


def evaluate_ship_gate(measured: ShipMeasurements, gate: ShipGateSettings) -> list[GateResult]:  # A1-A5, A7, A7b, A8, A16 and the recall rule
    precision_ok = measured.pair_precision is not None and measured.pair_precision >= gate.pair_precision_minimum  # A8 needs a threshold
    results = [  # one gate result per rule, in report order
        check_gate("A1", measured.facts_right >= gate.facts_minimum and measured.turned_wrong_count <= gate.facts_maximum_turned_wrong,  # A1: enough facts right and few turned wrong
                   "%d of 30 right (min %d), %d turned wrong (max %d)" % (measured.facts_right, gate.facts_minimum, measured.turned_wrong_count,  # A1 detail: counts against the limits
                                                                            gate.facts_maximum_turned_wrong)),  # A1 detail: the turned-wrong limit
        check_gate("A2", measured.facts_without_pairs_right >= gate.facts_without_pairs_minimum,  # A2: enough facts right among the questions that are not stored pairs
                   "%d of 27 right (min %d)" % (measured.facts_without_pairs_right, gate.facts_without_pairs_minimum)),  # A2 detail: count against the minimum
        check_gate("A3", measured.known_ten_right >= gate.known_ten_minimum, "%d of 10 right (min %d)" % (measured.known_ten_right, gate.known_ten_minimum)),  # A3: enough of the known ten right
        check_gate("A4", not measured.iron_throne_names_mountain, "Iron Throne answers name Gregor, Clegane or the Mountain: %s" % measured.iron_throne_names_mountain),  # A4: no Iron Throne answer names the wrong man
        check_gate("A5", measured.test_hit_at_5 >= gate.passage_hit_at_5_minimum, "hit-at-5 %.3f (min %.2f)" % (measured.test_hit_at_5, gate.passage_hit_at_5_minimum)),  # A5: passage found in the top five often enough
        check_gate("A7", measured.unanswerable_evidence_share <= gate.unanswerable_evidence_maximum  # A7: few unanswerable questions get evidence
                   and measured.development_evidence_share >= gate.development_evidence_minimum,  # A7: and enough development questions keep it
                   "unanswerable with evidence %.3f (max %.2f), development with evidence %.3f (min %.2f)" % (  # A7 detail: both shares against their limits
                       measured.unanswerable_evidence_share, gate.unanswerable_evidence_maximum, measured.development_evidence_share,  # A7 detail: unanswerable and development shares
                       gate.development_evidence_minimum)),  # A7 detail: development minimum
        check_gate("A7b", measured.fact_development_evidence_share >= gate.fact_development_evidence_minimum,  # A7b: fact-development questions keep evidence
                   "%.3f (min %.2f)" % (measured.fact_development_evidence_share, gate.fact_development_evidence_minimum)),  # A7b detail: share against the minimum
        check_gate("A8", precision_ok, "held-back pair precision %s (min %.2f)" % (measured.pair_precision, gate.pair_precision_minimum)),  # A8: stored pairs are precise enough, if any are shown
        check_gate("A16", measured.copyright_problem_count == 0, "%d copyright findings over all answers" % measured.copyright_problem_count),  # A16: no copyright finding in any answer
        check_gate("key_term_recall", measured.chosen_key_term_recall >= measured.baseline_key_term_recall,  # recall rule: the chosen arm finds at least as many key terms as the baseline
                   "chosen %.3f against question_only %.3f" % (measured.chosen_key_term_recall, measured.baseline_key_term_recall)),  # recall detail: chosen against baseline
    ]
    results += [GateResult(name, "live_check", "measured on the Space after the deploy") for name in ("A13", "A14", "A15")]  # not measurable on the Mac
    return results  # A9-A12 are likewise confirmed live


def build_recommended_settings(decisions: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:  # the file the integrator copies into service_settings.py
    return {**decisions, "evidence": dict(evidence)}  # chosen values first, then the numbers behind each
