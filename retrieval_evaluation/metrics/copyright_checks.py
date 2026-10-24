# A16: no response field may share more than 25 consecutive words with ANY passage of the bundle (not only the packed ones), at most one
# book snippet per response, each snippet at most 25 words. Written independently of the service's runtime guard, on purpose (DESIGN.md §11.2).
import re  # word pattern
from typing import Iterable, Mapping, Sequence  # inputs

MAX_SHARED_WORDS = 25  # more consecutive shared words than this is a copy
FORBIDDEN_RUN_LENGTH = MAX_SHARED_WORDS + 1  # a shared run of this many words proves a copy longer than the limit
MAX_SNIPPET_WORDS = 25  # longest allowed book snippet
BOOK_SOURCE_TYPE = "book_passage"  # source_type of a passage source
COPYRIGHT_WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")  # lower-case words with inner apostrophes
ELLIPSIS = "…"  # marks a snippet that was cut


def split_copyright_words(text: str) -> list[str]:  # lower-case words, curly apostrophes made straight
    return COPYRIGHT_WORD_PATTERN.findall(text.lower().replace("’", "'"))  # reading order


def hash_word_runs(words: Sequence[str], run_length: int) -> Iterable[int]:  # one hash per window of run_length consecutive words
    for start in range(len(words) - run_length + 1):  # every window start
        yield hash(tuple(words[start:start + run_length]))  # tuple hash (stable inside one process, which is all that is needed)


def build_passage_run_hashes(passage_texts: Iterable[str]) -> frozenset[int]:  # the 26-word run hashes of EVERY passage of a bundle
    run_hashes: set[int] = set()  # all hashes
    for passage_text in passage_texts:  # every passage, packed or not
        run_hashes.update(hash_word_runs(split_copyright_words(passage_text), FORBIDDEN_RUN_LENGTH))  # add its runs
    return frozenset(run_hashes)  # read-only set


def count_copied_runs(text: str, passage_run_hashes: frozenset[int]) -> int:  # how many 26-word windows of the text occur in some passage
    return sum(1 for run_hash in hash_word_runs(split_copyright_words(text), FORBIDDEN_RUN_LENGTH) if run_hash in passage_run_hashes)  # 0 = fine


def find_citation_problems(sources: Sequence[Mapping[str, object]]) -> list[str]:  # snippet rules of the public sources (no text in the messages)
    problems = []  # findings
    snippets = [source.get("snippet") for source in sources if source.get("source_type") == BOOK_SOURCE_TYPE and source.get("snippet")]  # book snippets
    if len(snippets) > 1:  # more than one book snippet in one response
        problems.append("%d book snippets in one response" % len(snippets))  # the rule is one
    for snippet in snippets:  # every book snippet
        word_count = len(split_copyright_words(str(snippet).replace(ELLIPSIS, " ")))  # words without the cut marker
        if word_count > MAX_SNIPPET_WORDS:  # too long
            problems.append("book snippet of %d words" % word_count)  # report only the length
    return problems  # empty when the sources obey the rules


def find_copyright_problems(completion: str, sources: Sequence[Mapping[str, object]], passage_run_hashes: frozenset[int]) -> list[str]:
    """Every A16 violation of one response: copied runs in the completion or in any snippet, and the snippet count and length rules."""
    problems = find_citation_problems(sources)  # snippet count and length
    texts = {"completion": completion}  # fields that carry text, by name
    texts.update({"snippet %d" % number: str(source["snippet"]) for number, source in enumerate(sources) if source.get("snippet")})  # every snippet
    for field_name, text in texts.items():  # every text field of the response
        copied_runs = count_copied_runs(text, passage_run_hashes)  # windows found in a passage
        if copied_runs:  # a copy longer than 25 words
            problems.append("%s shares more than %d words with a passage (%d windows)" % (field_name, MAX_SHARED_WORDS, copied_runs))  # numbers only
    return problems  # empty = the response passes
