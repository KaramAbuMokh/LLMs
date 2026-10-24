# G3: 40 questions the books cannot answer. 25 are unrelated to the books, 15 sound like the series but only the television show has the event.
# They must get no evidence (retrieval_used false, or a best score under the floor): the evidence gate's false-alarm test (A7).
from retrieval_evaluation.golden_sets.golden_question_records import ABSENT_KIND, UNRELATED_KIND, NoAnswerQuestion  # record and kinds

UNRELATED_QUESTIONS = (  # nothing to do with the books
    "How many moons does Jupiter have?", "What is the capital of Australia?", "Who wrote the play Hamlet?",
    "How do you make a sourdough starter?", "What is the boiling point of water at sea level?", "Who won the football World Cup in 2014?",
    "How many players are on a basketball team?", "What is the tallest mountain on Earth?", "How do I change a flat tire on a bicycle?",
    "What does DNA stand for?", "Who painted the Mona Lisa?", "What is the speed of light?",
    "Which programming language is best for beginners?", "What is the population of Canada?", "How many legs does a spider have?",
    "What is the chemical symbol for sodium?", "Who was the first person to walk on the Moon?", "What is the square root of 144?",
    "How long does it take to fly from London to New York?", "What is the currency of Japan?", "What causes the tides in the ocean?",
    "Who invented the telephone?", "What is the largest ocean on Earth?", "How many keys does a piano have?",
    "What is the best way to learn to swim?",
)
ABSENT_QUESTIONS = (  # sound like the series, but the events exist only in the television show or nowhere
    "Who kills the Night King?", "Which dragon does the Night King bring back from the dead?", "Who burns King's Landing in the final season?",
    "Who becomes king of the Six Kingdoms at the very end?", "Who stabs Daenerys Targaryen in the throne room?",
    "Who is beheaded in front of Daenerys at the walls of King's Landing?", "How does Cersei Lannister die?", "Who melts the Iron Throne?",
    "Who feeds Walder Frey a pie made of his sons?", "Who kills Ramsay Bolton in the Battle of the Bastards?",
    "Who rescues Jon Snow beyond the Wall on dragonback?", "Who kills the High Sparrow?", "Which actor plays Tyrion Lannister?",
    "Who wins the Battle of Winterfell against the army of the dead?", "Who is crowned king at the Dragonpit?",
)


def build_no_answer_questions() -> tuple[NoAnswerQuestion, ...]:  # G3 in a fixed order: unrelated first, then absent
    unrelated = [NoAnswerQuestion("g3-%02d" % number, question, UNRELATED_KIND) for number, question in enumerate(UNRELATED_QUESTIONS, start=1)]  # 25
    absent = [NoAnswerQuestion("g3-%02d" % number, question, ABSENT_KIND)  # 15, numbered on from the unrelated ones
              for number, question in enumerate(ABSENT_QUESTIONS, start=len(UNRELATED_QUESTIONS) + 1)]
    return tuple(unrelated + absent)  # 40 questions
