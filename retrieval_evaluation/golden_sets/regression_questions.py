# G4: the questions the old model got wrong (reported only; run once on the baseline and once on the chosen configuration). A4 reads the Iron Throne ones.
from retrieval_evaluation.golden_sets.golden_question_records import RegressionQuestion  # record
from retrieval_evaluation.metrics.fact_rule_loading import FactRule  # question plus word rules

IRON_THRONE_REJECTED_TERMS = ("gregor", "clegane", "mountain")  # the old model's wrong answer to who sits on the Iron Throne
REGRESSION_RULES = (  # (question, accepted groups, rejected terms)
    ("Who sits on the Iron Throne?", (), IRON_THRONE_REJECTED_TERMS),  # no accepted word: only the Mountain must not appear
    ("Who sits on the Iron Throne at the end of A Storm of Swords?", (("tommen",),), IRON_THRONE_REJECTED_TERMS + ("stannis",)),  # Tommen is king
    ("Who is called the Kingslayer?", (("jaime",),), ("stannis",)),  # Jaime Lannister
    ("Where is Theon Greyjoy's family from?", (("iron islands", "pyke", "ironborn"),), ()),  # same rule as the fact set
)


def build_regression_questions() -> tuple[RegressionQuestion, ...]:  # G4 in a fixed order
    return tuple(  # one record per rule
        RegressionQuestion("g4-%02d" % number, FactRule(question, accepted_groups, rejected_terms))  # the rule in the shared format
        for number, (question, accepted_groups, rejected_terms) in enumerate(REGRESSION_RULES, start=1)  # numbered from 1
    )


def is_iron_throne_question(question: RegressionQuestion) -> bool:  # True for the two questions that A4 reads
    return "Iron Throne" in question.rule.question  # by the question text
