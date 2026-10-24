# How a record becomes the text that is embedded, lexically indexed and reranked, and the chapter label shown in public citations
from retrieval_contracts.bundle_records import PassageRecord, QuestionAnswerRecord  # the stored records

INDEXED_TEXT_VERSION = 1  # stored in the manifest; any change to the functions below needs a rebuilt index
UNNUMBERED_CHAPTER_TITLES = frozenset({"Prologue", "Epilogue"})  # shown without a chapter number


def format_chapter_label(chapter_number: int | None, chapter_title: str | None) -> str | None:  # "Tyrion (chapter 12)", "Prologue" or None
    if chapter_title is None:  # the chapter is unknown
        return None  # citations then show the book only
    if chapter_number is None or chapter_number == 0 or chapter_title in UNNUMBERED_CHAPTER_TITLES:  # no meaningful number
        return chapter_title  # title alone
    return "%s (chapter %d)" % (chapter_title, chapter_number)  # point-of-view name with its reading-order number


def compose_passage_heading(record: PassageRecord) -> str:  # "A Storm of Swords, Tyrion (chapter 12)": the no-model version of contextual retrieval
    chapter_label = format_chapter_label(record.chapter_number, record.chapter_title)  # may be None
    return record.book_title if chapter_label is None else "%s, %s" % (record.book_title, chapter_label)  # book, then chapter when known


def compose_indexed_passage_text(record: PassageRecord) -> str:  # what the passage vector, its lexical postings and the reranker read
    return "%s: %s" % (compose_passage_heading(record), record.text)  # heading in front of the raw text


def compose_indexed_pair_text(record: QuestionAnswerRecord) -> str:  # what a pair's lexical postings read (its vector reads the question alone)
    return "%s %s" % (record.question, record.answer)  # question and answer, so names in either one count
