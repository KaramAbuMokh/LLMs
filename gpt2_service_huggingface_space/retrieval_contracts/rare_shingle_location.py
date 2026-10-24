# Where a question-answer text sits in the book parts, found by rare 4-word shingles (runs of 4 words that occur at most 3 times in all parts).
# ONE rule for the index builder (pair chapters, DESIGN.md §8.5) and the evaluation (gold spans and relevance of held-out questions, §11.1),
# so both agree on "where the answer is". Offsets always point into the ORIGINAL part text (words are matched without changing its length).
import re  # word pattern on the original text
from collections import Counter  # corpus-wide shingle counts
from dataclasses import dataclass  # frozen results
from typing import Mapping  # part texts by part number

SHINGLE_WORD_COUNT = 4  # words per shingle
RARE_SHINGLE_MAX_HITS = 3  # a shingle is evidence only when it occurs at most this often across all seven parts
MINIMUM_CLUSTER_HITS = 2  # a location needs two or more rare hits close together (98% agreement with coarse locations, probe 2026-10-24)
CLUSTER_WIDTH_CHARACTERS = 1000  # every hit of one cluster starts within this many characters of the cluster's first hit
LOCATION_WORD_PATTERN = re.compile(r"[a-z0-9]+(?:['’‘ʼ][a-z0-9]+)*", re.IGNORECASE | re.ASCII)  # ASCII words, inner apostrophes kept
CURLY_APOSTROPHE_TABLE = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})  # curly apostrophes become straight ones (length unchanged)


@dataclass(frozen=True)
class ShingleHit:  # one rare shingle of the searched text found in a part text
    part_number: int  # part the hit is in
    character_start: int  # first character of the shingle's first word
    character_end: int  # one past the last character of its fourth word


@dataclass(frozen=True)
class LocatedSpan:  # the densest cluster of rare hits: where the searched text most likely came from
    part_number: int  # part of every hit
    character_start: int  # first character of the first hit
    character_end: int  # one past the end of the last-ending hit
    hits: tuple[ShingleHit, ...]  # the cluster's hits by start (a passage "contains the answer" when it holds one of them whole)


@dataclass(frozen=True)
class SearchWindow:  # a stretch the hits must lie in (a pair's coarse ±20,000-character location); None searches every part
    part_number: int  # the one part searched
    character_start: int  # first character searched
    character_end: int  # one past the last character searched


@dataclass(frozen=True)
class RareShingleIndex:  # every rare shingle of the seven parts with the places it occurs (built once, about 1 GB for the real books)
    rare_hits: Mapping[str, tuple[ShingleHit, ...]]  # shingle text -> its 1 to RARE_SHINGLE_MAX_HITS places; common shingles are absent


def list_shingles(text: str) -> list[tuple[str, int, int]]:  # (shingle text, start, end) for every run of 4 words, in reading order
    words = [(match.group(0).lower().translate(CURLY_APOSTROPHE_TABLE), match.start(), match.end())  # folded word with its span
             for match in LOCATION_WORD_PATTERN.finditer(text)]  # every word of the original text
    return [  # one entry per window of SHINGLE_WORD_COUNT words
        (" ".join(word for word, _, _ in words[index:index + SHINGLE_WORD_COUNT]), words[index][1], words[index + SHINGLE_WORD_COUNT - 1][2])
        for index in range(len(words) - SHINGLE_WORD_COUNT + 1)  # every window start
    ]


def build_rare_shingle_index(part_texts: Mapping[int, str]) -> RareShingleIndex:  # counts every shingle, keeps the places of the rare ones
    shingles_by_part = {part_number: list_shingles(text) for part_number, text in part_texts.items()}  # every part's shingles
    corpus_counts = Counter(shingle for shingles in shingles_by_part.values() for shingle, _, _ in shingles)  # occurrences across all parts
    places: dict[str, list[ShingleHit]] = {}  # rare shingle -> its places
    for part_number, shingles in shingles_by_part.items():  # second pass, part by part
        for shingle, start, end in shingles:  # every shingle in reading order
            if corpus_counts[shingle] <= RARE_SHINGLE_MAX_HITS:  # rare enough to be evidence
                places.setdefault(shingle, []).append(ShingleHit(part_number, start, end))  # remember where
    return RareShingleIndex({shingle: tuple(hits) for shingle, hits in places.items()})  # frozen places


def is_inside_window(hit: ShingleHit, search_window: SearchWindow | None) -> bool:  # True when the hit lies wholly in the window
    if search_window is None:  # corpus-wide search
        return True  # every hit counts
    return (hit.part_number == search_window.part_number and search_window.character_start <= hit.character_start  # same part, starts inside
            and hit.character_end <= search_window.character_end)  # and ends inside


def locate_text(index: RareShingleIndex, text: str, search_window: SearchWindow | None = None) -> LocatedSpan | None:
    """Finds the densest cluster of the text's rare shingles; None when no MINIMUM_CLUSTER_HITS hits cluster (ties: earliest part and start)."""
    searched_shingles = {shingle for shingle, _, _ in list_shingles(text)}  # each distinct shingle of the text counts once
    hits = sorted(  # candidate hits by part, then start
        (hit for shingle in searched_shingles for hit in index.rare_hits.get(shingle, ()) if is_inside_window(hit, search_window)),
        key=lambda hit: (hit.part_number, hit.character_start, hit.character_end),  # reading order, deterministic
    )
    best_cluster: tuple[ShingleHit, ...] = ()  # densest cluster so far
    window_end = 0  # one past the last hit of the current cluster (two-pointer scan)
    for first_position, first_hit in enumerate(hits):  # each hit opens a cluster
        window_end = max(window_end, first_position + 1)  # the cluster holds at least its own first hit
        while (window_end < len(hits) and hits[window_end].part_number == first_hit.part_number  # same part
               and hits[window_end].character_start - first_hit.character_start <= CLUSTER_WIDTH_CHARACTERS):  # close enough
            window_end += 1  # take the next hit into the cluster
        if window_end - first_position > len(best_cluster):  # strictly denser than every earlier cluster
            best_cluster = tuple(hits[first_position:window_end])  # keep it
    if len(best_cluster) < MINIMUM_CLUSTER_HITS:  # one stray rare phrase is not a location
        return None  # unlocated
    return LocatedSpan(  # the cluster as a span
        best_cluster[0].part_number, best_cluster[0].character_start, max(hit.character_end for hit in best_cluster), best_cluster,
    )
