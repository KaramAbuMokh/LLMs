# Public citations for an answer: book + chapter per source, and ONE book snippet of at most 25 words per answer (copyright policy)
import re  # word spans
from dataclasses import dataclass  # frozen record
from gpt2_demo_service.grounded_answers.context_packing import PackedPrompt  # the evidence the prompt holds
from retrieval_contracts.bundle_records import SourceType  # the two source kinds
from retrieval_contracts.indexed_text import format_chapter_label  # "Tyrion (chapter 12)"

MAX_SNIPPET_WORDS = 25  # no public field shows more than this many words of any text
WORD_SPAN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")  # a word is letters and digits with inner apostrophes: the same rule as the evaluation's independent copyright check, so a "25 word" snippet never counts as 26 or more words there (dashes split words)
ELLIPSIS = "…"  # marks a cut snippet


@dataclass(frozen=True)
class SourceCitation:  # one source shown to the visitor
    source_type: SourceType  # book passage or question-answer pair
    book: str | None  # book title, None when the source was never located
    chapter: str | None  # chapter label, None when unknown
    snippet: str | None  # at most 25 words: the opening of the first passage, or a pair's question


def limit_words(text: str, max_words: int) -> str:  # the first max_words words of the text, with an ellipsis when cut
    word_spans = list(WORD_SPAN_PATTERN.finditer(text))  # every word with its position
    if len(word_spans) <= max_words:  # short enough already
        return text.strip()  # unchanged
    return text[:word_spans[max_words - 1].end()] + ELLIPSIS  # cut after the last allowed word


def choose_book_snippet(passage_text: str, max_words: int = MAX_SNIPPET_WORDS) -> str:  # the passage opening; takes no question, so a caller cannot aim it
    return limit_words(passage_text, max_words)  # passages start at a sentence start


def build_source_citations(packed: PackedPrompt, book_snippet_enabled: bool) -> tuple[SourceCitation, ...]:  # passages best first, then pairs
    citations = []  # one per packed source
    for position, passage in enumerate(packed.passages):  # best passage first
        record = passage.record  # the stored passage
        snippet = choose_book_snippet(record.text) if position == 0 and book_snippet_enabled else None  # only the first passage shows book text
        citations.append(SourceCitation(SourceType.BOOK_PASSAGE, record.book_title, format_chapter_label(record.chapter_number, record.chapter_title), snippet))  # one source
    for pair in packed.pair_examples:  # then the solved examples
        record = pair.record  # the stored pair
        citations.append(SourceCitation(  # a pair's snippet is its question, written by another model (not book text)
            SourceType.QUESTION_ANSWER_PAIR, record.book_title, format_chapter_label(record.chapter_number, record.chapter_title),
            limit_words(record.question, MAX_SNIPPET_WORDS),
        ))
    return tuple(citations)  # in display order
