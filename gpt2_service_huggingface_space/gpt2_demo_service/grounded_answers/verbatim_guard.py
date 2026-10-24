# Copy guard: cuts an answer at the first stretch that repeats more than 25 consecutive words of the books, whichever route made the answer
import re  # sentence ends
from dataclasses import dataclass  # frozen result
from gpt2_demo_service.grounded_answers.book_run_index import BookRunIndex, list_words  # runs of every book passage, and the shared word rule

SENTENCE_END_PATTERN = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s|$)")  # punctuation that closes a sentence
ELLIPSIS = "…"  # marks a cut inside a sentence


@dataclass(frozen=True)  # immutable record
class GuardedCompletion:  # the answer after the guard
    text: str  # the answer, cut when it copied too much
    was_cut: bool  # True when text is shorter than the input


def limit_verbatim_overlap(completion: str, book_runs: BookRunIndex | None) -> GuardedCompletion:  # cuts the first stretch longer than the allowed words; without an index nothing can be checked
    if book_runs is None:  # the books are not loaded (retrieval off or its start failed)
        return GuardedCompletion(completion, False)  # nothing to compare with
    completion_words = list_words(completion)  # the answer's words with their places
    cut_word_index = book_runs.find_first_copied_word_index([word for word, _, _ in completion_words])  # the word that makes a run longer than allowed
    if cut_word_index is None:  # no stretch of the books is repeated
        return GuardedCompletion(completion, False)  # unchanged
    cut_position = completion_words[cut_word_index][1]  # where the offending word starts
    sentence_ends = [match.end() for match in SENTENCE_END_PATTERN.finditer(completion[:cut_position])]  # sentence ends before it
    if sentence_ends:  # cut back to the last whole sentence
        return GuardedCompletion(completion[:sentence_ends[-1]].rstrip(), True)  # sentence-aligned cut
    allowed_end = completion_words[cut_word_index - 1][2]  # end of the last allowed shared word
    return GuardedCompletion(completion[:allowed_end] + ELLIPSIS, True)  # no sentence end: keep the allowed words and mark the cut
