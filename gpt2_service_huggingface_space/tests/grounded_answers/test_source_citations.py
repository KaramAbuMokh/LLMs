# Tests of the public citations: one question-independent book snippet, other passages without text, pair snippets are questions, 25-word limit
import random  # seeded invented text
import re  # independent word count in the new test
from gpt2_demo_service.grounded_answers.answer_settings import PromptLayout  # layout names
from gpt2_demo_service.grounded_answers.context_packing import PromptBudget, pack_prompt  # builds packed prompts
from gpt2_demo_service.grounded_answers.source_citations import MAX_SNIPPET_WORDS, build_source_citations, choose_book_snippet, limit_words  # code under test
from retrieval_contracts.bundle_records import SourceType  # source kinds


def pack(evidence_factory, tokenizer, question, passages, pairs):  # packs with a roomy budget and the combined layout
    evidence = evidence_factory["evidence"](passages, pairs)  # invented evidence
    return pack_prompt(question, evidence, PromptLayout.PASSAGES_AND_PAIR_EXAMPLES, PromptBudget(1024, 120, 700, 8), tokenizer)  # packed prompt


def test_exactly_one_book_snippet_on_the_first_passage(evidence_factory, gpt2_tokenizer):  # three passages and an example, one snippet
    passages = [evidence_factory["passage"]("bp-%d" % number, evidence_factory["text"](random.Random(number), 4), 0, number * 10_000) for number in range(3)]  # three passages
    pair = evidence_factory["pair"]("qa-1", "Who carried the lantern?", "Brannoc did.")  # one example
    citations = build_source_citations(pack(evidence_factory, gpt2_tokenizer, "Q?", passages, [pair]), True)  # cite
    kinds = [citation.source_type for citation in citations]  # source kinds in order
    assert kinds == [SourceType.BOOK_PASSAGE] * len([k for k in kinds if k == SourceType.BOOK_PASSAGE]) + [SourceType.QUESTION_ANSWER_PAIR]  # passages first, then the pair
    book_snippets = [c.snippet for c in citations if c.source_type == SourceType.BOOK_PASSAGE and c.snippet is not None]  # snippets of book text
    assert len(book_snippets) == 1 and citations[0].snippet == book_snippets[0]  # only the first passage shows book text


def test_snippet_is_the_same_whatever_the_question(evidence_factory, gpt2_tokenizer):  # a caller cannot aim the snippet
    passage = evidence_factory["passage"]("bp-1", evidence_factory["text"](random.Random(9), 5), 0, 0)  # one passage
    first = build_source_citations(pack(evidence_factory, gpt2_tokenizer, "Who carried the lantern?", [passage], []), True)  # one question
    second = build_source_citations(pack(evidence_factory, gpt2_tokenizer, "Where is the tower?", [passage], []), True)  # another question
    assert first[0].snippet == second[0].snippet == choose_book_snippet(passage.record.text)  # the passage opening in both cases


def test_snippets_have_at_most_25_words(evidence_factory, gpt2_tokenizer):  # book snippets and pair snippets
    long_question = " ".join(["word"] * 60) + "?"  # a pair question of 60 words
    passage = evidence_factory["passage"]("bp-1", evidence_factory["text"](random.Random(8), 6), 0, 0)  # a long passage
    citations = build_source_citations(pack(evidence_factory, gpt2_tokenizer, "Q?", [passage], [evidence_factory["pair"]("qa-1", long_question, "A.")]), True)  # citations of a long passage and a long pair question
    assert all(len(citation.snippet.split()) <= MAX_SNIPPET_WORDS for citation in citations)  # the limit holds for both kinds
    assert citations[0].snippet.endswith("…") and citations[-1].snippet.endswith("…")  # both were cut and say so


def test_pair_snippet_is_its_question_and_switch_turns_book_text_off(evidence_factory, gpt2_tokenizer):  # book snippet switch
    passage = evidence_factory["passage"]("bp-1", evidence_factory["text"](random.Random(7), 3), 0, 0)  # one passage
    pair = evidence_factory["pair"]("qa-1", "Who carried the lantern?", "Brannoc did.")  # one example
    citations = build_source_citations(pack(evidence_factory, gpt2_tokenizer, "Q?", [passage], [pair]), False)  # snippets off
    assert citations[0].snippet is None and citations[1].snippet == "Who carried the lantern?"  # pair question stays, book text goes


def test_chapter_label_and_missing_chapter(evidence_factory, gpt2_tokenizer):  # "Mira (chapter 1)" or None, the field is still present
    located = evidence_factory["passage"]("bp-1", "A sentence.", 0, 0, (1, "Mira"))  # chapter known
    unlocated = evidence_factory["passage"]("bp-2", "Another sentence.", 0, 5000, (None, None))  # chapter unknown
    citations = build_source_citations(pack(evidence_factory, gpt2_tokenizer, "Q?", [located, unlocated], []), True)  # cite both
    assert (citations[0].book, citations[0].chapter) == ("The Glass Coast", "Mira (chapter 1)") and citations[1].chapter is None  # labels


def test_limit_words_keeps_short_text_and_marks_cuts():  # the helper behind both snippet kinds
    assert limit_words("  Short text.  ", 25) == "Short text."  # unchanged apart from stripping
    assert limit_words("one two three four", 2) == "one two…"  # cut after the second word


def test_dash_joined_words_count_as_separate_words():  # a snippet of "word--word" runs must not pass 25 words under the stricter count (found by the evaluation's copyright check)
    dashed = " ".join("alpha—beta" for _ in range(20))  # 20 whitespace words, 40 real words
    snippet = limit_words(dashed, MAX_SNIPPET_WORDS)  # cut at 25 real words
    assert snippet.endswith("…") and len(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", snippet.replace("…", " ").lower())) == MAX_SNIPPET_WORDS  # exactly 25 words by the independent count
