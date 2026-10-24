# Raw heading readers on synthetic raw lines: contents lists with glued names, split-letter headings in prose-style books, numbered chapters with lost lines
from retrieval_index_builder.source_alignment.heading_reading.body_heading_scanning import scan_body_headings, spell_number  # body-style readers
from retrieval_index_builder.source_alignment.heading_reading.contents_list_reading import read_contents_headings  # contents-style reader
from retrieval_index_builder.source_alignment.heading_reading.raw_chapter_headings import clean_raw_line, read_raw_chapter_headings  # public reader

PROSE = "The long grey road wound over the hills and down into the sleeping valley below the keep."  # a prose line: too long to be a heading


def test_storm_style_contents_split_glued_names_and_stop_at_the_appendix():  # "CATELYNJON" -> two headings, "SANSAEPILOGUE" -> two, appendix titles dropped
    lines = ["Contents", "Title Page", "PROLOGUE", "JAIME", "CATELYN", "ARYA", "TYRION", "DAVOS", "SANSA", "JON", "DAENERYS", "BRAN", "CATELYNJON", "SANSAEPILOGUE",  # a contents list followed by the appendix titles
             "THE KINGS AND THEIR COURTS", "THE KING ON THE IRON THRONE", PROSE]  # a contents list followed by the appendix titles
    assert read_contents_headings(lines) == ["PROLOGUE", "JAIME", "CATELYN", "ARYA", "TYRION", "DAVOS", "SANSA", "JON", "DAENERYS", "BRAN", "CATELYN", "JON", "SANSA", "EPILOGUE"]  # expected headings: glued names split, appendix dropped


def test_game_style_contents_split_mixed_case_glue():  # "EddardTyrion" -> two names, upper-cased
    lines = ["Contents", "Prologue", "Bran", "Catelyn", "Daenerys", "Eddard", "Jon", "Arya", "Tyrion", "Sansa", "Bran", "EddardTyrion", "Appendix", "House Stark", PROSE]  # a mixed-case contents list
    assert read_contents_headings(lines) == ["PROLOGUE", "BRAN", "CATELYN", "DAENERYS", "EDDARD", "JON", "ARYA", "TYRION", "SANSA", "BRAN", "EDDARD", "TYRION"]  # expected headings: upper case, glued names split


def test_a_prologue_line_followed_by_prose_is_not_a_contents_list():  # Dance and Clash have no contents list
    assert read_contents_headings(["PROLOGU E", "The", PROSE, PROSE, PROSE, PROSE, PROSE, PROSE, PROSE, PROSE, PROSE, PROSE]) == []  # nothing to read


def test_dance_style_body_headings_include_glued_tails_and_stop_at_the_epilogue():  # whole-capital lines and "prose. NAME" tails, nothing after EPILOGUE
    lines = ["A TITLE LINE", PROSE, "April 2011 PROLOGUE", PROSE, "TYRION", PROSE, "She nodded and left. BRAN", PROSE, "THE LOST LORD", PROSE, "EPILOGUE", "WESTEROS", "THE BOY KING"]  # a body with a glued tail and an epilogue followed by the appendix
    assert scan_body_headings(lines) == ["PROLOGUE", "TYRION", "BRAN", "THE LOST LORD", "EPILOGUE"]  # expected headings


def test_clash_style_chapters_fill_lost_numbers_and_lost_names():  # CHAPTER TWO lost entirely, THREE lost its name line
    lines = ["PROLOGU E", "The", PROSE, "CHAPTER ONE", "ARYA", PROSE, "CHAPTER THREE", PROSE, "CHAPTER FOUR", "BRAN", "THE KING ON THE IRON THRONE", PROSE]  # numbered chapters with lost lines
    assert scan_body_headings(lines) == ["PROLOGUE", "CHAPTER ONE ARYA", "CHAPTER TWO", "CHAPTER THREE", "CHAPTER FOUR BRAN"]  # expected headings with lost numbers filled in


def test_number_words_cover_one_to_ninety_nine():  # the spelling used for the Clash chapter numbers
    assert [spell_number(number) for number in (1, 12, 19, 20, 21, 30, 69, 99)] == ["ONE", "TWELVE", "NINETEEN", "TWENTY", "TWENTY-ONE", "THIRTY", "SIXTY-NINE", "NINETY-NINE"]  # expected number words


def test_reader_picks_the_contents_list_when_there_is_one_and_cleans_tabs(tmp_path):  # file-level behaviour
    raw_lines = ["Contents", "Prologue", "Bran", "Catelyn", "Daenerys", "Eddard", "Jon", "Arya", "Tyrion", "Sansa", "Bran", "Jon", "Appendix", PROSE]  # a mixed-case contents list for the file-level reader
    raw_path = tmp_path / "raw_book.txt"  # a raw book file
    raw_path.write_text("\n".join("\t" + line + " \t" for line in raw_lines), encoding="utf-8")  # tabs and spaces around every line
    assert read_raw_chapter_headings(raw_path)[:3] == ["PROLOGUE", "BRAN", "CATELYN"]  # contents list read
    assert clean_raw_line("\tAME\tOF\t") == "AME OF"  # tabs fold into single spaces
