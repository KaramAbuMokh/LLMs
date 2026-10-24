# Free data fixes from the 10-24 data review (section 4 item 3), each a switch, applied to TRAINING pairs only
# Validation pairs stay untouched so the current and the new model are scored on exactly the same questions.
import re  # word patterns
from collections import Counter, defaultdict  # word frequencies and the near-duplicate index

# Questions that point at the hidden 200-character chunk ("in the passage") cannot be answered from memory
CITES_HIDDEN_SOURCE = re.compile(r"\b(the|this|that) (chunk|passage|text|excerpt|snippet)\b|\bmentioned in the\b", re.IGNORECASE)
UNNAMED_SUBJECT = re.compile(  # "the character", "the speaker", "the man": nobody is named
    r"\bthe (main )?(character|characters|speaker|narrator|protagonist|man|woman|boy|girl|person|individual|individuals)\b",
    re.IGNORECASE,
)
INTERPRETIVE_TAIL = re.compile(  # ", indicating that ..." endings the books never say (group 1 keeps a closing quote)
    r",(['\"\u2019\u201d]?)\s*(indicating|suggesting|highlighting|showing|emphasizing|reflecting|demonstrating|revealing|underscoring|implying)\b[^.]*\.?$",
    re.IGNORECASE,
)
WORD = re.compile(r"[a-z']+")  # lowercase words
# Filler words ignored when comparing pairs for near-duplicates
STOP_WORDS = set("the a an and or of to in on at for with by from is was are were be been his her their its he she they it that this "
                 "what who whom whose which when where why how does did do has have had as about into than then them him".split())
NEAR_DUPLICATE_SIMILARITY = 0.6  # mean of question and answer word overlap at or above this = near-duplicate
RARE_KEYS_PER_PAIR = 3  # a pair is compared with earlier pairs that share one of its 3 rarest words
MAX_POSTINGS_PER_KEY = 300  # words shared by more pairs than this are too common to use as keys


def content_words(text):  # set of meaningful lowercase words
    return {word for word in WORD.findall(text.lower()) if len(word) > 2 and word not in STOP_WORDS}  # no filler words


def overlap(first, second):  # Jaccard similarity of two word sets
    return len(first & second) / max(1, len(first | second))  # shared share of all words


def has_named_entity(question):  # a capitalised name or place after the first word
    return any(word[:1].isupper() and word != "I" for word in question.split()[1:])  # "Jon", "Winterfell", ...


class NearDuplicateFinder:  # remembers kept pairs and spots new ones that repeat them
    def __init__(self, pairs):  # word frequencies over every pair decide which words are rare
        self.frequency = Counter(word for pair in pairs for word in content_words(pair["q"] + " " + pair["a"]))  # document counts
        self.postings, self.kept = defaultdict(list), []  # rare word -> kept pair numbers, and the kept word sets

    def is_near_duplicate(self, question, answer):  # True when an already kept pair says the same thing; else remembers this one
        question_words, answer_words = content_words(question), content_words(answer)  # word sets
        keys = sorted(question_words | answer_words, key=lambda word: (self.frequency[word], word))[:RARE_KEYS_PER_PAIR]  # rarest, ties by spelling
        # earlier kept pairs that share one of those rare words (very common keys are skipped)
        candidates = {number for key in keys if len(self.postings[key]) < MAX_POSTINGS_PER_KEY for number in self.postings[key]}
        for number in candidates:  # earlier kept pairs sharing a rare word
            kept_question, kept_answer = self.kept[number]  # their word sets
            # mean of question overlap and answer overlap at or above 0.6 = the same fact again
            if (overlap(question_words, kept_question) + overlap(answer_words, kept_answer)) / 2 >= NEAR_DUPLICATE_SIMILARITY:
                return True  # same fact again
        for key in keys:  # index this pair under its rare words
            self.postings[key].append(len(self.kept))  # its number among kept pairs
        self.kept.append((question_words, answer_words))  # its word sets
        return False  # new fact


def drop_reason(question, answer, settings, seen_questions, near_duplicates):  # first switch that rejects the pair, or None
    if not question or not answer:  # nothing to learn
        return "empty"  # always dropped
    if settings.drop_unnamed_subject_pairs and UNNAMED_SUBJECT.search(question):  # cannot be answered from memory
        return "unnamed_subject"  # review: 10,014 on the full file
    if settings.drop_source_citing_pairs and CITES_HIDDEN_SOURCE.search(question):  # "in the passage"
        return "cites_hidden_source"  # review: 8,800
    if settings.drop_pairs_without_named_entity and not has_named_entity(question):  # no name or place at all
        return "no_named_entity"  # review: 5,316
    if settings.drop_near_duplicate_pairs and near_duplicates.is_near_duplicate(question, answer):  # repeats a kept pair
        return "near_duplicate"  # review: 1,556
    if settings.drop_repeated_questions and " ".join(question.lower().split()) in seen_questions:  # exact repeat of a question
        return "repeated_question"  # review: 1,416
    if len(answer.split()) < settings.min_answer_words:  # too short to carry a fact with context
        return "short_answer"  # review: 686 under 8 words
    return None  # keep


def filter_training_pairs(pairs, settings):  # returns the kept pairs and how many each switch removed
    removed, seen_questions, kept, tails_cut = Counter(), set(), [], 0  # report counters, memory, result
    near_duplicates = NearDuplicateFinder(pairs)  # built once over the training pairs
    for pair in pairs:  # in source order, so the first copy of a repeat is kept
        question, answer = pair["q"].strip(), pair["a"].strip()  # trimmed text
        reason = drop_reason(question, answer, settings, seen_questions, near_duplicates)  # filter verdict
        if reason:  # dropped
            removed[reason] += 1  # count the reason
            continue  # next pair
        seen_questions.add(" ".join(question.lower().split()))  # remember the kept question
        if settings.cut_interpretive_tails and INTERPRETIVE_TAIL.search(answer):  # ", indicating ..." ending
            answer, tails_cut = INTERPRETIVE_TAIL.sub(r"\1.", answer), tails_cut + 1  # end the sentence before the tail
        kept.append({**pair, "q": question, "a": answer})  # cleaned pair
    report = {"input": len(pairs), "kept": len(kept), "removed": dict(removed), "interpretive_tails_cut": tails_cut}  # summary
    return kept, report  # pairs and the report
