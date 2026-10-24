# Sentence splitting and overlong-sentence cutting on invented text
from retrieval_index_builder.passage_chunking.gpt2_token_counting import count_gpt2_tokens  # token unit
from retrieval_index_builder.passage_chunking.sentence_spans import cut_overlong_spans, split_sentence_spans  # code under test


def split_texts(text: str) -> list[str]:  # the sentences as strings
    return [text[start:end] for start, end in split_sentence_spans(text)]  # slices by span


def test_sentences_split_after_end_marks_and_closing_quotes_when_a_capital_follows():  # dialogue and plain prose
    text = "  “Not here. Not now.” The rider turned. Was it late? It was! Then he went on... and on.  "  # leading and trailing spaces too
    assert split_texts(text) == ["“Not here.", "Not now.”", "The rider turned.", "Was it late?", "It was!", "Then he went on... and on."]  # six sentences


def test_no_split_before_a_lowercase_word_or_inside_a_speech_tag():  # "?” he said" continues the sentence
    assert split_texts("“Where?” he asked. “Home.”") == ["“Where?” he asked.", "“Home.”"]  # speech tag stays attached
    assert split_texts("It cost 3.5 gold, no more. Fine.") == ["It cost 3.5 gold, no more.", "Fine."]  # a decimal point is not a boundary


def test_spans_are_exact_slices_without_the_spaces_between_sentences():  # offsets point into the original text
    text = "One two three. Four five six. Seven eight nine."  # three sentences
    assert split_sentence_spans(text) == [(0, 14), (15, 29), (30, 47)]  # offsets


def test_empty_and_blank_text_have_no_sentences():  # edge cases
    assert split_sentence_spans("") == [] and split_sentence_spans("   ") == []  # nothing to split


def test_overlong_sentence_is_cut_at_word_boundaries_into_pieces_under_the_target():  # one sentence of 600 words
    text = " ".join("word%d" % number for number in range(600)) + "."  # a single huge sentence
    spans = split_sentence_spans(text)  # one span
    pieces = cut_overlong_spans(text, spans, max_tokens=230, target_tokens=100)  # code under test
    assert len(pieces) > 5 and all(count_gpt2_tokens(text[start:end]) <= 100 for start, end in pieces)  # all under the target
    assert pieces[0][0] == 0 and pieces[-1][1] == len(text)  # nothing lost at the ends
    assert all(text[start] != " " and text[end - 1] != " " for start, end in pieces)  # cut at word boundaries
    assert all(first[1] < second[0] for first, second in zip(pieces, pieces[1:]))  # pieces do not overlap


def test_sentences_under_the_ceiling_are_left_alone():  # no cutting needed
    text = "A short one. Another short one."  # two sentences
    spans = split_sentence_spans(text)  # their spans
    assert cut_overlong_spans(text, spans, max_tokens=230, target_tokens=200) == spans  # unchanged
