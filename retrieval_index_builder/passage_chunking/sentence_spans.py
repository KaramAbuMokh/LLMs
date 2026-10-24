# Finds sentence boundaries in cleaned book text and cuts the rare sentence that is longer than a passage may be
import re  # boundary and word patterns
from typing import Sequence  # span lists
from retrieval_index_builder.passage_chunking.gpt2_token_counting import count_gpt2_tokens  # passage size unit

SENTENCE_BOUNDARY_PATTERN = re.compile(r"([.!?…]+[\"”’')\]]*)(\s+)(?=[\"“‘'(\[—]?[A-Z0-9])")  # end mark, closing quotes, spaces, then a capital or opening quote
WORD_PATTERN = re.compile(r"\S+")  # words for cutting an overlong sentence


def split_sentence_spans(text: str) -> list[tuple[int, int]]:  # (start, end) of every sentence, spaces between sentences left out
    spans = []  # sentence spans in reading order
    sentence_start = len(text) - len(text.lstrip())  # skip leading spaces
    for boundary in SENTENCE_BOUNDARY_PATTERN.finditer(text):  # every sentence end
        if boundary.end(1) > sentence_start:  # a non-empty sentence
            spans.append((sentence_start, boundary.end(1)))  # ends after its closing quote
        sentence_start = boundary.end(2)  # the next sentence starts after the spaces
    text_end = len(text.rstrip())  # trailing spaces do not belong to the last sentence
    if text_end > sentence_start:  # a last sentence without an end mark
        spans.append((sentence_start, text_end))  # keep it
    return spans  # offsets inside `text`


def cut_overlong_spans(text: str, spans: Sequence[tuple[int, int]], max_tokens: int, target_tokens: int) -> list[tuple[int, int]]:  # cuts sentences that cannot fit a passage
    cut_spans = []  # spans none of which exceeds max_tokens
    for sentence_start, sentence_end in spans:  # every sentence
        if count_gpt2_tokens(text[sentence_start:sentence_end]) <= max_tokens:  # fits in a passage
            cut_spans.append((sentence_start, sentence_end))  # keep it whole
            continue  # next sentence
        piece_start = sentence_start  # the piece being grown word by word
        previous_word_end = sentence_start  # end of the last word that still fits
        for word in WORD_PATTERN.finditer(text, sentence_start, sentence_end):  # words of the long sentence
            if previous_word_end > piece_start and count_gpt2_tokens(text[piece_start:word.end()]) > target_tokens:  # this word would overfill the piece
                cut_spans.append((piece_start, previous_word_end))  # close the piece before it
                piece_start = word.start()  # the next piece starts with this word
            previous_word_end = word.end()  # the word is inside the open piece
        cut_spans.append((piece_start, sentence_end))  # the last piece
    return cut_spans  # pseudo-sentences that all fit a passage
