# Plans where passages start and end inside one stretch of text: whole sentences, a token target, a hard ceiling, sentence overlap, no tiny tail
from typing import Sequence  # span lists
from retrieval_index_builder.index_build_settings import IndexBuildSettings  # sizes and overlap rules
from retrieval_index_builder.passage_chunking.gpt2_token_counting import count_gpt2_tokens  # passage size unit
from retrieval_index_builder.passage_chunking.sentence_spans import cut_overlong_spans, split_sentence_spans  # sentences


def choose_overlap_start(previous_range: tuple[int, int] | None, position: int, token_counts: Sequence[int], settings: IndexBuildSettings) -> int:  # where the next passage starts: a repeated sentence or the next new one
    if previous_range is None or settings.overlap_sentences == 0:  # first passage, or overlap switched off
        return position  # starts with a new sentence
    overlap_start = max(position - settings.overlap_sentences, previous_range[0] + 1)  # last sentences of the previous passage, never all of it
    overlap_tokens = sum(token_counts[overlap_start:position])  # their size
    if overlap_start >= position or overlap_tokens > settings.maximum_overlap_tokens:  # nothing to repeat, or too long to repeat
        return position  # no overlap
    if overlap_tokens + token_counts[position] > settings.max_tokens:  # the overlap would leave no room for new text
        return position  # no overlap
    return overlap_start  # repeat these sentences first


def exact_token_count(part_text: str, sentence_spans: Sequence[tuple[int, int]], first_index: int, end_index: int) -> int:  # tokens of the joined sentences
    return count_gpt2_tokens(part_text[sentence_spans[first_index][0]:sentence_spans[end_index - 1][1]])  # real count, spaces between sentences included


def plan_sentence_ranges(part_text: str, sentence_spans: Sequence[tuple[int, int]], settings: IndexBuildSettings) -> list[tuple[int, int]]:  # which sentences form which passage
    token_counts = [count_gpt2_tokens(part_text[start:end]) for start, end in sentence_spans]  # size of each sentence alone
    ranges: list[tuple[int, int]] = []  # (first sentence, one past the last) of each passage
    position = 0  # first sentence not yet in any passage
    while position < len(sentence_spans):  # until every sentence is covered
        first_index = choose_overlap_start(ranges[-1] if ranges else None, position, token_counts, settings)  # where this passage starts
        end_index, running_tokens = position + 1, sum(token_counts[first_index:position + 1])  # always at least one new sentence
        while end_index < len(sentence_spans) and running_tokens + token_counts[end_index] <= settings.target_tokens:  # room left under the target
            running_tokens += token_counts[end_index]  # take the next sentence
            end_index += 1  # advance
        while exact_token_count(part_text, sentence_spans, first_index, end_index) > settings.max_tokens:  # joined text is over the ceiling
            if end_index - position > 1:  # more than one new sentence
                end_index -= 1  # give the last one back
            else:  # a single new sentence plus its overlap is too long
                first_index = position  # drop the overlap (one sentence always fits)
                break  # done shrinking
        ranges.append((first_index, end_index))  # the passage
        position = end_index  # continue after it
    if len(ranges) > 1 and exact_token_count(part_text, sentence_spans, *ranges[-1]) < settings.minimum_tail_tokens:  # tiny last passage
        merged_range = (ranges[-2][0], ranges[-1][1])  # previous passage extended to the end
        if exact_token_count(part_text, sentence_spans, *merged_range) <= settings.max_tokens:  # still under the ceiling
            ranges[-2:] = [merged_range]  # one passage instead of two
    return ranges  # sentence index ranges


def plan_passage_spans(part_text: str, stretch_start: int, stretch_end: int, settings: IndexBuildSettings) -> list[tuple[int, int]]:  # character spans of the passages
    local_spans = split_sentence_spans(part_text[stretch_start:stretch_end])  # sentences relative to the stretch
    sentence_spans = cut_overlong_spans(part_text, [(stretch_start + start, stretch_start + end) for start, end in local_spans], settings.max_tokens, settings.target_tokens)  # sentence spans in part-text offsets, overlong ones cut
    ranges = plan_sentence_ranges(part_text, sentence_spans, settings)  # which sentences form which passage
    return [(sentence_spans[first_index][0], sentence_spans[end_index - 1][1]) for first_index, end_index in ranges]  # offsets in the part text
