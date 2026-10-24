# Word-overlap scores for model answers: token F1, key-term recall against a gold answer, and invented-name detection (DESIGN.md §11.2)
import re  # word and name patterns
from collections import Counter  # multiset overlap
from typing import Sequence  # lists of texts
from retrieval_contracts.lexical_tokens import STOPWORDS, normalize_word, split_lexical_tokens  # the contract's word rules

OVERLAP_WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")  # lower-case words with inner apostrophes
CAPITALISED_WORD_PATTERN = re.compile(r"(?<![A-Za-z0-9'’])([A-Z][a-z]+(?:['’][a-z]+)*)")  # a word that starts with a capital letter
SENTENCE_END_CHARACTERS = ".!?:\n"  # a capital after one of these (or at the start) is just a sentence start


def split_overlap_tokens(text: str) -> list[str]:  # lower-case words of a text, curly apostrophes made straight
    return OVERLAP_WORD_PATTERN.findall(text.lower().replace("’", "'"))  # reading order, repeats kept


def normalize_question_text(text: str) -> str:  # one comparable form of a question (exact-match checks against the pair questions)
    return " ".join(split_overlap_tokens(text))  # punctuation and case gone


def token_f1_score(candidate: str, reference: str) -> float:  # F1 of the shared word multiset (1.0 for two empty texts)
    candidate_tokens, reference_tokens = split_overlap_tokens(candidate), split_overlap_tokens(reference)  # both word lists
    if not candidate_tokens or not reference_tokens:  # one side has no words
        return 1.0 if candidate_tokens == reference_tokens else 0.0  # equal only when both are empty
    shared_count = sum((Counter(candidate_tokens) & Counter(reference_tokens)).values())  # words in common, repeats counted once each
    if shared_count == 0:  # nothing in common
        return 0.0  # no overlap
    precision, recall = shared_count / len(candidate_tokens), shared_count / len(reference_tokens)  # both overlap shares
    return 2 * precision * recall / (precision + recall)  # harmonic mean


def key_term_recall(answer: str, gold_answer: str, question: str) -> float | None:
    """Share of the gold answer's key terms (lexical tokens that the question does not already contain) found in the answer; None without any."""
    gold_terms = set(split_lexical_tokens(gold_answer)) - set(split_lexical_tokens(question))  # new information in the gold answer
    if not gold_terms:  # the gold answer adds nothing to the question
        return None  # such a question is left out of the mean
    return len(gold_terms & set(split_lexical_tokens(answer))) / len(gold_terms)  # found share


def find_capitalised_names(text: str) -> list[str]:  # capitalised words that are not just the start of a sentence or a function word
    names = []  # kept names, in order
    for match in CAPITALISED_WORD_PATTERN.finditer(text):  # every capitalised word
        before = text[:match.start()].rstrip()  # text in front of it
        starts_sentence = not before or before[-1] in SENTENCE_END_CHARACTERS  # first word of the text or of a sentence
        if starts_sentence and match.group(1).lower() in STOPWORDS:  # "The", "She", "It" at a sentence start
            continue  # not a name
        names.append(match.group(1))  # a name
    return names  # capitalised words in reading order


def find_unsupported_names(answer: str, supporting_texts: Sequence[str]) -> list[str]:
    """Names in the answer that appear in none of the supporting texts (the question plus the packed context): the 'Gregor on the throne' detector."""
    supported_words = {normalize_word(word) for text in supporting_texts for word in split_overlap_tokens(text)}  # every supported word
    return [name for name in find_capitalised_names(answer) if normalize_word(name.lower().replace("’", "'")) not in supported_words]  # unsupported
