# Memory of every stretch of book words: one 64-bit hash per run of 26 consecutive words across ALL passages, so the copy guard can spot a memorised stretch on any route
import re  # word pattern
from dataclasses import dataclass  # frozen index record
from typing import Mapping, Sequence  # inputs
import numpy  # arrays of hashes
from numpy.typing import NDArray  # array type hints
from retrieval_contracts.bundle_records import PassageRecord  # stored passages

FORBIDDEN_RUN_WORDS = 26  # a run of this many words shared with the books is longer than the 25 words any response may show
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")  # the same word rule as the evaluation's independent copyright check, so the guard is never blinder than the check
MAX_JOINABLE_GAP_CHARACTERS = 2  # passages this close in the part text read as one unbroken stretch
HASH_MULTIPLIER = numpy.uint64(0x9E3779B97F4A7C15)  # odd 64-bit constant of the rolling hash
WORD_VALUE_SEED = 20261025  # fixed seed, so the same books always give the same index


def list_words(text: str) -> list[tuple[str, int, int]]:  # (folded word, start, end) for every word
    return [(match.group(0).lower().replace("’", "'"), match.start(), match.end()) for match in WORD_PATTERN.finditer(text)]  # lower case, straight apostrophes


def join_adjacent_passages(passages: Sequence[PassageRecord]) -> list[list[str]]:  # word lists of unbroken stretches; neighbours share one sentence, which is counted once
    stretches: list[list[str]] = []  # one word list per stretch
    previous: PassageRecord | None = None  # the passage before the current one in the same part
    for passage in sorted(passages, key=lambda candidate: (candidate.part_number, candidate.character_start)):  # reading order
        new_text = passage.text  # the text not yet seen
        joins_previous = previous is not None and previous.part_number == passage.part_number and passage.character_start - previous.character_end <= MAX_JOINABLE_GAP_CHARACTERS  # touching or overlapping
        if joins_previous and passage.character_end <= previous.character_end:  # nothing new at all
            continue  # the stretch already holds it
        if joins_previous and passage.character_start < previous.character_end:  # the shared sentence comes first
            new_text = passage.text[previous.character_end - passage.character_start:]  # keep only what follows it
        if not joins_previous:  # a gap, another part or the first passage
            stretches.append([])  # a new stretch begins
        stretches[-1].extend(word for word, _, _ in list_words(new_text))  # add the new words
        previous = passage  # the next passage compares against this one
    return stretches  # every unbroken stretch of book words


def hash_word_runs(word_values: NDArray[numpy.uint64], run_length: int) -> NDArray[numpy.uint64]:  # one hash per window of run_length consecutive words
    window_count = len(word_values) - run_length + 1  # windows that fit
    if window_count <= 0:  # fewer words than one run
        return numpy.empty(0, dtype=numpy.uint64)  # nothing to hash
    hashes = numpy.zeros(window_count, dtype=numpy.uint64)  # running hashes, wrapping around at 2**64 on purpose
    for offset in range(run_length):  # one pass per word position inside the window
        hashes = hashes * HASH_MULTIPLIER + word_values[offset:offset + window_count]  # polynomial rolling hash
    return hashes  # one hash per window


@dataclass(frozen=True)  # immutable record
class BookRunIndex:  # read-only after construction: about 8 bytes per indexed run
    run_length: int  # words per run
    word_values: Mapping[str, numpy.uint64]  # random 64-bit value of every word that occurs in the books
    sorted_run_hashes: NDArray[numpy.uint64]  # unique run hashes, ascending

    @property  # read-only attribute
    def run_count(self) -> int:  # distinct runs held
        return len(self.sorted_run_hashes)  # size of the index

    def find_first_copied_word_index(self, words: Sequence[str]) -> int | None:  # index of the word that completes the first run found in the books (None when there is none)
        values = numpy.array([self.word_values.get(word, numpy.uint64(0)) for word in words], dtype=numpy.uint64)  # 0 marks a word no book contains
        if len(values) < self.run_length or len(self.sorted_run_hashes) == 0:  # too short, or an empty index
            return None  # nothing can match
        window_hashes = hash_word_runs(values, self.run_length)  # hash of every window of the text
        unknown_total = numpy.concatenate(([0], numpy.cumsum(values == 0)))  # running count of unknown words
        has_unknown_word = (unknown_total[self.run_length:] - unknown_total[:-self.run_length]) > 0  # a window with an unknown word cannot match
        places = numpy.minimum(numpy.searchsorted(self.sorted_run_hashes, window_hashes), len(self.sorted_run_hashes) - 1)  # where each hash would sit
        found = (self.sorted_run_hashes[places] == window_hashes) & ~has_unknown_word  # windows that are in the books
        first_window = numpy.flatnonzero(found)  # their starts
        return int(first_window[0]) + self.run_length - 1 if len(first_window) else None  # the last word of the first match


def build_book_run_index(passages: Sequence[PassageRecord], run_length: int = FORBIDDEN_RUN_WORDS) -> BookRunIndex:  # reads every passage once
    random_source = numpy.random.default_rng(WORD_VALUE_SEED)  # deterministic word values
    word_numbers: dict[str, int] = {}  # word -> its number
    stretch_arrays = [numpy.array([word_numbers.setdefault(word, len(word_numbers)) for word in words], dtype=numpy.int64) for words in join_adjacent_passages(passages)]  # numbers per stretch
    value_of_number = random_source.integers(1, numpy.iinfo(numpy.uint64).max, size=len(word_numbers), dtype=numpy.uint64)  # one random value per distinct word
    run_hashes = [hash_word_runs(value_of_number[numbers], run_length) for numbers in stretch_arrays]  # runs never cross a gap, because each stretch is hashed alone
    sorted_hashes = numpy.unique(numpy.concatenate(run_hashes)) if run_hashes else numpy.empty(0, dtype=numpy.uint64)  # unique and ascending
    return BookRunIndex(run_length, {word: value_of_number[number] for word, number in word_numbers.items()}, sorted_hashes)  # the finished index
