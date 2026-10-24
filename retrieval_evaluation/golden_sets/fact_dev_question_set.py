# FD: 42 new fact questions across all five books, used to choose settings (never reported as a result). Each is scored by the same word-rule
# judge as the 30 fact questions. Every entry stores WHERE the fact is stated (part number and character offset, no book text) so a script can
# check that an accepted term of each group really appears there (DESIGN.md §11.1). Parts: 0 Dance, 1 Storm, 2 Game, 3 Feast, 5 Clash.
from typing import Mapping, Sequence  # inputs of the checks
from retrieval_evaluation.golden_sets.golden_question_records import FactDevelopmentQuestion, FactQuestion  # records
from retrieval_evaluation.metrics.answer_overlap_scoring import normalize_question_text, token_f1_score  # question comparison
from retrieval_evaluation.metrics.fact_rule_loading import FactRule, FactRuleBook  # question plus word rules

FACT_DEVELOPMENT_ENTRIES = (  # (question, accepted groups, rejected terms, evidence part number, evidence character offset)
    ("Which Slaver's Bay city does Daenerys make her seat of rule?", (("meereen",),), (), 0, 79260),  # Dance: answer meereen
    ("Who commands the Unsullied for Daenerys?", (("grey worm",),), (), 0, 69784),  # Dance: answer grey worm
    ("Which blue-bearded sellsword captain serves Daenerys Targaryen?", (("daario",),), (), 0, 76386),  # Dance: answer daario
    ("Which house has a flayed man as its sigil?", (("bolton",),), (), 1, 87753),  # Storm: answer bolton
    ("Who is the Lord of the Crossing at the Twins?", (("walder", "frey"),), (), 1, 554692),  # Storm: answer walder
    ("Whom does Robb Stark secretly marry?", (("jeyne", "westerling"),), (), 1, 384462),  # Storm: answer jeyne
    ("Who kills Lord Tywin Lannister?", (("tyrion",),), (), 1, 2127854),  # Storm: answer tyrion
    ("What are the names of Daenerys Targaryen's three dragons?", (("drogon",), ("rhaegal",), ("viserion",)), (), 1, 212660),  # Storm: answer drogon
    ("What does valar morghulis mean?", (("all men must die",),), (), 1, 743175),  # Storm: answer all men must die
    ("Which prince of Dorne is nicknamed the Red Viper?", (("oberyn",),), (), 1, 1039351),  # Storm: answer oberyn
    ("What do the Martells of Dorne say as their motto?", (("unbowed",),), (), 1, 2274086),  # Storm: answer unbowed
    ("Which sword does Brienne receive from Jaime Lannister?", (("oathkeeper",),), (), 1, 2005259),  # Storm: answer oathkeeper
    ("Who is the King-beyond-the-Wall?", (("mance",),), (), 1, 190705),  # Storm: answer mance
    ("Which wildling girl does Jon Snow fall in love with?", (("ygritte",),), (), 1, 182125),  # Storm: answer ygritte
    ("Which knight is called the Knight of Flowers?", (("loras",),), (), 1, 331758),  # Storm: answer loras
    ("Which uncle of Catelyn Stark is called the Blackfish?", (("brynden",),), (), 1, 391241),  # Storm: answer brynden
    ("Which maester serves at Winterfell?", (("luwin",),), (), 2, 43139),  # Game: answer luwin
    ("Who teaches Arya Stark water dancing in King's Landing?", (("syrio",),), (), 2, 430117),  # Game: answer syrio
    ("How does Viserys Targaryen die?", (("molten gold", "gold*", "crown"),), (), 2, 976883),  # Game: answer molten gold
    ("Who is Daenerys Targaryen's older brother?", (("viserys",),), (), 2, 48255),  # Game: answer viserys
    ("What kind of tree stands in the godswood of Winterfell?", (("weirwood",),), (), 2, 36915),  # Game: answer weirwood
    ("Who is Jon Snow's uncle on the Wall?", (("benjen",),), (), 2, 85440),  # Game: answer benjen
    ("Which animal wounds King Robert Baratheon on his last hunt?", (("boar",),), (), 2, 142031),  # Game: answer boar
    ("Who beheads Eddard Stark?", (("ilyn", "payne"),), (), 2, 300726),  # Game: answer ilyn
    ("Who is the fat bookish friend of Jon Snow at the Wall?", (("sam", "samwell"),), (), 2, 503405),  # Game: answer sam
    ("Who is Lord Commander of the Kingsguard under King Robert?", (("barristan", "selmy"),), (), 2, 144541),  # Game: answer barristan
    ("Which eunuch serves as Master of Whisperers?", (("varys",),), (), 2, 209821),  # Game: answer varys
    ("What is the Dothraki word for the horde that follows a khal?", (("khalasar",),), (), 2, 57102),  # Game: answer khalasar
    ("Which greatsword does Eddard Stark carry?", (("ice",),), (), 2, 23927),  # Game: answer ice
    ("Which Stark child has the direwolf named Nymeria?", (("arya",),), (), 2, 132513),  # Game: answer arya
    ("How tall is the Wall?", (("seven hundred feet", "700 feet"),), (), 2, 350105),  # Game: answer seven hundred feet
    ("What is Littlefinger's real name?", (("petyr", "baelish"),), (), 2, 373859),  # Game: answer petyr
    ("Which Greyjoy uncle is the priest called Damphair?", (("aeron",),), (), 3, 35912),  # Feast: answer aeron
    ("To which city does Samwell Tarly travel to become a maester?", (("oldtown",),), (), 3, 12525),  # Feast: answer oldtown
    ("Whom is Brienne of Tarth searching for in A Feast for Crows?", (("sansa",),), (), 3, 127200),  # Feast: answer sansa
    ("Which disgraced maester serves Queen Cersei?", (("qyburn",),), (), 3, 116502),  # Feast: answer qyburn
    ("Which Greyjoy is called Crow's Eye?", (("euron",),), (), 3, 46205),  # Feast: answer euron
    ("Which red priestess serves Stannis Baratheon?", (("melisandre",),), (), 5, 35941),  # Clash: answer melisandre
    ("What burns the ships on the Blackwater?", (("wildfire",),), (), 5, 115257),  # Clash: answer wildfire
    ("Which smuggler knight is known as the Onion Knight?", (("davos",),), (), 5, 18669),  # Clash: answer davos
    ("What motto do the Greyjoys of Pyke live by?", (("do not sow",),), (), 5, 304897),  # Clash: answer do not sow
    ("Which god do the red priests serve?", (("r'hllor", "lord of light"),), (), 5, 37663),  # Clash: answer r'hllor
)


def build_fact_development_questions() -> tuple[FactDevelopmentQuestion, ...]:  # FD in a fixed order
    return tuple(  # one record per entry
        FactDevelopmentQuestion("fd-%02d" % number, FactRule(question, accepted_groups, rejected_terms), part_number, character_offset)  # numbered
        for number, (question, accepted_groups, rejected_terms, part_number, character_offset) in enumerate(FACT_DEVELOPMENT_ENTRIES, start=1)
    )


def find_overlap_problems(questions: Sequence[FactDevelopmentQuestion], fact_questions: Sequence[FactQuestion], maximum_f1: float) -> list[str]:
    """FD questions that are one of the 30 or closer to one of them than maximum_f1 token F1 (ids and numbers only)."""
    problems = []  # findings
    for question in questions:  # every FD question
        for fact_question in fact_questions:  # every one of the 30
            overlap = token_f1_score(question.rule.question, fact_question.rule.question)  # word overlap
            if overlap > maximum_f1:  # too close
                problems.append("%s has token F1 %.2f with %s" % (question.question_id, overlap, fact_question.question_id))  # report
    return problems  # empty = all clear


def find_pair_duplicate_problems(questions: Sequence[FactDevelopmentQuestion], pair_questions: Sequence[str]) -> list[str]:  # word-for-word pair questions
    stored_questions = {normalize_question_text(question) for question in pair_questions}  # comparable forms of the 34,001 pair questions
    return ["%s is word for word a pair question" % question.question_id for question in questions  # one finding per duplicate
            if normalize_question_text(question.rule.question) in stored_questions]


def find_evidence_problems(questions: Sequence[FactDevelopmentQuestion], rule_book: FactRuleBook, part_texts: Mapping[int, str],
                           window_characters: int) -> list[str]:  # every accepted group must have a term near the evidence offset
    problems = []  # findings
    for question in questions:  # every FD question
        part_text = part_texts.get(question.evidence_part_number)  # the cleaned part
        if part_text is None:  # unknown part number
            problems.append("%s names part %d which does not exist" % (question.question_id, question.evidence_part_number))  # report
            continue  # nothing to read
        offset = question.evidence_character_offset  # where the fact is stated
        nearby_text = part_text[max(0, offset - window_characters):offset + window_characters]  # the text around it
        for group_number, group in enumerate(question.rule.accepted_groups, start=1):  # every accepted group
            if not any(rule_book.term_occurs(term, nearby_text) for term in group):  # no term of the group appears
                problems.append("%s: no term of group %d near its evidence offset" % (question.question_id, group_number))  # report
    return problems  # empty = every fact is stated where claimed
