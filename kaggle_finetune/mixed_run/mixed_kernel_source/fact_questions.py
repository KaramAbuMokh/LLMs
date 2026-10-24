# The 30 fact questions with known answers (the 10 fixed smoke questions first) and the word rules that mark an answer right.
# A term matches as a whole word ("ned" does not match "owned"); a trailing * matches any ending ("kill*" = kill, killed, killing).
# Right = every accepted group has a match and no rejected term appears. The same rules score all three models.
import re  # word matching
from run_settings import FIXED_QUESTIONS  # the 10 questions of the smoke check and of this morning's report

KILLED = ["kill*", "murder*", "slain", "slaughter*", "massacre*", "butcher*", "betray*", "death", "died", "dead"]  # Red Wedding verbs
FACT_QUESTIONS = (  # (question, accepted groups, rejected terms)
    (FIXED_QUESTIONS[0], [["daenerys", "dany", "stormborn"]], []),  # mother of dragons
    (FIXED_QUESTIONS[1], [["winter is coming"]], []),  # Stark words
    (FIXED_QUESTIONS[2], [["dwarf*", "imp", "halfman", "tywin*"]], []),  # Tyrion
    (FIXED_QUESTIONS[3], [["wall", "defend*", "protect*", "guard*", "wildling*", "the others", "white walkers"]], []),  # Night's Watch
    (FIXED_QUESTIONS[4], [["olenna", "littlefinger", "baelish", "queen of thorns"]], []),  # Joffrey's poisoners in the books
    (FIXED_QUESTIONS[5], [["needle"]], ["longclaw"]),  # Arya's sword
    (FIXED_QUESTIONS[6], [["stable*", "simple*", "servant", "giant", "huge", "carr* bran", "big man"]], ["direwolf", "wolf"]),  # Hodor
    (FIXED_QUESTIONS[7], [["robb", "catelyn", "king in the north", "starks"], KILLED], []),  # Red Wedding
    (FIXED_QUESTIONS[8], [["warrior", "fighter", "swordswoman", "rainbow guard", "renly", "selwyn"]], ["wife", "beaut*"]),  # Brienne
    (FIXED_QUESTIONS[9], [["regent", "arrest*", "eddard", "ned", "faith", "sparrow*", "wildfire", "robert"]], []),  # Cersei
    ("What is the name of Jon Snow's direwolf?", [["ghost"]], []),
    ("What is the name of Jon Snow's sword?", [["longclaw"]], []),
    ("What is the seat of House Stark?", [["winterfell"]], []),
    ("What is the seat of House Lannister?", [["casterly rock"]], []),
    ("What are the words of House Lannister?", [["hear me roar"]], []),
    ("What are the words of House Targaryen?", [["fire and blood"]], []),
    ("What is the sigil of House Lannister?", [["lion*"]], []),
    ("What is the sigil of House Baratheon?", [["stag*"]], []),
    ("Who did King Robert name Hand of the King after Jon Arryn died?", [["eddard", "ned"]], []),
    ("What is the name of Bran Stark's direwolf?", [["summer"]], []),
    ("What is the name of Robb Stark's direwolf?", [["grey wind"]], []),
    ("Who is Daenerys Targaryen's husband in A Game of Thrones?", [["drogo"]], []),
    ("Who is the Lord Commander of the Night's Watch when Jon Snow joins it?", [["mormont", "jeor"]], []),
    ("Who is Tyrion Lannister's father?", [["tywin"]], []),
    ("What is the capital city of the Seven Kingdoms?", [["king's landing", "kings landing"]], []),
    ("Who is called the Kingslayer?", [["jaime"]], []),
    ("What is Ser Gregor Clegane called?", [["mountain"]], []),
    ("What is Sandor Clegane called?", [["hound"]], []),
    ("Where is Theon Greyjoy's family from?", [["iron islands", "pyke", "ironborn"]], []),
    ("What can kill the Others?", [["dragonglass", "obsidian", "valyrian steel"]], []),
)


def term_pattern(term):  # whole-word regex for one term; * = any word ending; spaces may be any whitespace
    words = [re.escape(word[:-1]) + r"\w*" if word.endswith("*") else re.escape(word) + r"\b" for word in term.split()]  # per word
    return re.compile(r"\b" + r"\s+".join(words))  # "carr* bran" matches "carried Bran" and "carries bran"


def is_right(answer, accepted_groups, rejected_terms):  # True when every group matches and no rejected term appears
    text = answer.lower().replace("’", "'")  # lower case, curly apostrophes made straight
    matches = lambda term: term_pattern(term).search(text) is not None  # one term found in the answer
    return all(any(matches(term) for term in group) for group in accepted_groups) and not any(matches(term) for term in rejected_terms)


def score_fact_answers(sampled):  # sampled = [{"question", "samples": [{"answer", ...}]}] in FACT_QUESTIONS order
    rules = {question: (accepted, rejected) for question, accepted, rejected in FACT_QUESTIONS}  # question -> its rules
    majority_right, samples_right = 0, 0  # counters
    for record in sampled:  # every question
        accepted, rejected = rules[record["question"]]  # its rules
        for sample in record["samples"]:  # every sampled answer
            sample["right"] = is_right(sample["answer"], accepted, rejected)  # mark it
        right_count = sum(sample["right"] for sample in record["samples"])  # right answers for this question
        record["majority_right"] = right_count * 2 > len(record["samples"])  # 2 of 3
        majority_right, samples_right = majority_right + record["majority_right"], samples_right + right_count  # add up
    return {"facts_majority_right": majority_right, "facts_questions": len(sampled), "facts_samples_right": samples_right,
            "facts_samples": sum(len(record["samples"]) for record in sampled)}  # the facts score
