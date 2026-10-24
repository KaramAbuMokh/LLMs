# Cuts the chapters (and the chapterless stretches between them) of the book parts into stored PassageRecords
from dataclasses import dataclass  # frozen stretch description
from typing import Mapping, Sequence  # inputs
from retrieval_contracts.book_parts import find_book_part  # book title and series number
from retrieval_contracts.bundle_records import ChapterRecord, PassageRecord  # stored record types
from retrieval_index_builder.index_build_settings import IndexBuildSettings  # sizes and overlap rules
from retrieval_index_builder.passage_chunking.gpt2_token_counting import count_gpt2_tokens  # token_count of each passage
from retrieval_index_builder.passage_chunking.passage_boundary_planning import plan_passage_spans  # where passages start and end

CHUNKER_VERSION = 2  # stored in the manifest; any change to the chunking rules needs a new number and a rebuilt index (2: text after the last chapter is titled "Appendix")
UNKNOWN_CHAPTER_ID_PART = "xxx"  # chapter slot of a passage id when the chapter is unknown
APPENDIX_TITLE = "Appendix"  # chapter title of the text after the last chapter of a part (public citations then say "Appendix" instead of showing no chapter)
HEADING_GAP_CHARACTERS = 120  # a gap this short between chapters is just a heading (and page-number debris), not text worth indexing


@dataclass(frozen=True)  # immutable record
class TextStretch:  # a stretch of a part text outside every chapter (appendix, or a whole book that failed the alignment gate)
    part_number: int  # part it lies in
    character_start: int  # first character
    character_end: int  # one past the last character
    title: str | None  # "Appendix" for the tail after the last chapter, None for any other stretch


def build_passage_records(part_text: str, part_number: int, chapter: ChapterRecord | None, span: tuple[int, int], first_chunk_index: int,  # one chapter (or one chapterless span) cut into stored passage records
                          settings: IndexBuildSettings, stretch_title: str | None = None) -> list[PassageRecord]:  # passages of one chapter or one chapterless stretch (stretch_title names a stretch)
    book_part = find_book_part(part_number)  # book of this part
    chapter_slot = UNKNOWN_CHAPTER_ID_PART if chapter is None else "%03d" % chapter.chapter_number  # id slot
    passage_spans = plan_passage_spans(part_text, span[0], span[1], settings)  # where passages start and end
    return [  # one record per passage
        PassageRecord(  # the stored passage
            passage_id="bp-%d-%s-%03d" % (part_number, chapter_slot, first_chunk_index + chunk_index), part_number=part_number,  # unique id and part
            book_title=book_part.book_title, series_number=book_part.series_number,  # book of the part
            chapter_number=chapter.chapter_number if chapter else None, chapter_title=chapter.chapter_title if chapter else stretch_title,  # chapter, if known; else the stretch's own title
            character_start=start, character_end=end, token_count=count_gpt2_tokens(part_text[start:end]), text=part_text[start:end],  # span, size, exact text
        )
        for chunk_index, (start, end) in enumerate(passage_spans)  # passages in reading order
    ]


def chunk_chapter(part_text: str, chapter: ChapterRecord, settings: IndexBuildSettings) -> list[PassageRecord]:  # passages of one chapter, never crossing its edges
    return build_passage_records(part_text, chapter.part_number, chapter, (chapter.character_start, chapter.character_end), 0, settings)  # numbered from 0


def list_part_pieces(part_number: int, part_length: int, part_chapters: Sequence[ChapterRecord]) -> list[ChapterRecord | TextStretch]:  # chapters and the gaps between
    pieces: list[ChapterRecord | TextStretch] = []  # pieces in text order
    covered_until = 0  # how far the part is covered
    for chapter in sorted(part_chapters, key=lambda record: record.character_start):  # chapters in text order
        if chapter.character_start - covered_until > HEADING_GAP_CHARACTERS:  # real text before the chapter that no chapter owns
            pieces.append(TextStretch(part_number, covered_until, chapter.character_start, None))  # a chapterless stretch
        pieces.append(chapter)  # the chapter itself
        covered_until = chapter.character_end  # continue after it
    if part_length - covered_until > HEADING_GAP_CHARACTERS:  # text after the last chapter (appendix) or a whole unaligned part
        pieces.append(TextStretch(part_number, covered_until, part_length, APPENDIX_TITLE if part_chapters else None))  # tail: an appendix when chapters came before it, else a whole unaligned part
    return pieces  # cover the whole part except heading gaps


def chunk_all_parts(part_texts: Mapping[int, str], chapters: Sequence[ChapterRecord], settings: IndexBuildSettings) -> tuple[PassageRecord, ...]:  # cuts every part of every book
    passages: list[PassageRecord] = []  # all passages, part by part
    for part_number in sorted(part_texts):  # every part
        part_text = part_texts[part_number]  # its text
        chapterless_count = 0  # chapterless passages so far, so their ids stay unique inside the part
        for piece in list_part_pieces(part_number, len(part_text), [chapter for chapter in chapters if chapter.part_number == part_number]):  # in text order
            if isinstance(piece, ChapterRecord):  # a chapter
                passages.extend(chunk_chapter(part_text, piece, settings))  # cut it
            else:  # a chapterless stretch
                stretch_passages = build_passage_records(part_text, part_number, None, (piece.character_start, piece.character_end), chapterless_count, settings, piece.title)  # cut it
                chapterless_count += len(stretch_passages)  # chapterless ids continue across stretches
                passages.extend(stretch_passages)  # keep them
    return tuple(passages)  # in reading order
