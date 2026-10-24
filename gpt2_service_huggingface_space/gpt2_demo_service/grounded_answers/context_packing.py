# Packs retrieved evidence into the trained prompt within a hard token budget: best items whole, only the last one trimmed
import re  # sentence ends
from dataclasses import dataclass  # frozen records
import tiktoken  # GPT-2 tokenizer
from gpt2_demo_service.grounded_answers.answer_settings import PromptLayout  # layout names
from gpt2_demo_service.grounded_answers.prompt_layouts import PAIR_LAYOUTS, PASSAGE_LAYOUTS, render_pair_example, render_prompt  # prompt text
from gpt2_demo_service.index_search.evidence_records import RetrievedEvidence, ScoredPair, ScoredPassage  # what retrieval found

MINIMUM_TRIMMED_TOKENS = 40  # a trimmed last passage shorter than this is dropped instead
SEPARATOR_TOKEN_ESTIMATE = 2  # the blank line between two prompt blocks
MAX_OVERLAP_SHARE = 0.5  # a passage is skipped when more than this share of the shorter one overlaps a chosen one
SENTENCE_END_PATTERN = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s|$)")  # punctuation that closes a sentence (with closing quotes)


@dataclass(frozen=True)
class PromptBudget:  # the token limits of one prompt
    context_length: int  # the model's context window (1,024)
    answer_reserve_tokens: int  # tokens kept free for the answer
    context_token_budget: int  # most tokens of retrieved text
    safety_margin_tokens: int  # extra free tokens


@dataclass(frozen=True)
class PackedPrompt:  # the finished prompt and the evidence it really holds
    layout: PromptLayout  # layout used
    prompt_text: str  # what the model reads
    prompt_token_ids: tuple[int, ...]  # the same text as GPT-2 tokens
    passages: tuple[ScoredPassage, ...]  # passages included, best first (the last one may be trimmed)
    pair_examples: tuple[ScoredPair, ...]  # examples included, best first (never trimmed)

    @property
    def has_context(self) -> bool:  # True when any evidence reached the prompt
        return bool(self.passages or self.pair_examples)  # either kind counts


def count_tokens(text: str, tokenizer: tiktoken.Encoding) -> int:  # GPT-2 tokens of a text, special words treated as plain text
    return len(tokenizer.encode(text, disallowed_special=()))  # length of the encoding


def trim_to_sentence_budget(text: str, max_tokens: int, tokenizer: tiktoken.Encoding) -> str:  # longest prefix that ends at a sentence end and fits
    if max_tokens <= 0:  # no room at all
        return ""  # nothing fits
    if count_tokens(text, tokenizer) <= max_tokens:  # already fits
        return text  # unchanged
    best_prefix = ""  # longest fitting prefix so far
    for sentence_end in SENTENCE_END_PATTERN.finditer(text):  # every sentence end, in reading order
        prefix = text[:sentence_end.end()]  # text up to this sentence end
        if count_tokens(prefix, tokenizer) > max_tokens:  # too long from here on
            break  # later prefixes only grow
        best_prefix = prefix  # fits
    return best_prefix  # may be empty when even the first sentence is too long


def overlaps_chosen_passage(candidate: ScoredPassage, chosen: list[ScoredPassage]) -> bool:  # True when most of the shorter passage is already covered
    for kept in chosen:  # passages already taken
        if kept.record.part_number != candidate.record.part_number:  # different part, cannot overlap
            continue  # next one
        overlap = min(kept.record.character_end, candidate.record.character_end) - max(kept.record.character_start, candidate.record.character_start)  # shared characters
        shorter = min(kept.record.character_end - kept.record.character_start, candidate.record.character_end - candidate.record.character_start)  # shorter length
        if overlap > MAX_OVERLAP_SHARE * shorter:  # adjacent chunks share one sentence and stay allowed
            return True  # skip the candidate
    return False  # no heavy overlap


def select_items(evidence: RetrievedEvidence, layout: PromptLayout, allowance: int, tokenizer: tiktoken.Encoding
                 ) -> tuple[list[ScoredPair], list[ScoredPassage], list[str]]:  # pair examples, passages and the passage texts to render
    pairs: list[ScoredPair] = []  # examples taken
    passages: list[ScoredPassage] = []  # passages taken
    passage_texts: list[str] = []  # rendered passage texts (the last may be trimmed)
    remaining = allowance  # tokens still free
    for pair in (evidence.pair_examples if layout in PAIR_LAYOUTS else ()):  # best example first
        cost = count_tokens(render_pair_example(pair.record.question, pair.record.answer), tokenizer) + SEPARATOR_TOKEN_ESTIMATE  # tokens it needs
        if cost > remaining:  # an example is all or nothing: a cut answer would teach the model a broken example
            break  # lower-ranked examples are not tried
        pairs.append(pair)  # take it
        remaining -= cost  # spend the tokens
    for passage in (evidence.passages if layout in PASSAGE_LAYOUTS else ()):  # best passage first
        if overlaps_chosen_passage(passage, passages):  # mostly a copy of a chosen one
            continue  # skip it
        cost = count_tokens(passage.record.text, tokenizer) + SEPARATOR_TOKEN_ESTIMATE  # tokens it needs
        if cost <= remaining:  # fits whole
            passages.append(passage)  # take it
            passage_texts.append(passage.record.text)  # raw text
            remaining -= cost  # spend the tokens
            continue  # next passage
        trimmed_text = trim_to_sentence_budget(passage.record.text, remaining - SEPARATOR_TOKEN_ESTIMATE, tokenizer)  # cut at a sentence end
        if count_tokens(trimmed_text, tokenizer) >= MINIMUM_TRIMMED_TOKENS:  # still worth showing
            passages.append(passage)  # take the trimmed passage
            passage_texts.append(trimmed_text)  # its trimmed text
        break  # the first passage that does not fit ends packing
    return pairs, passages, passage_texts  # selection in priority order


def pack_prompt(question: str, evidence: RetrievedEvidence, layout: PromptLayout, budget: PromptBudget, tokenizer: tiktoken.Encoding) -> PackedPrompt:  # builds the prompt and the list of evidence it really holds
    base_ids = tokenizer.encode(render_prompt(PromptLayout.QUESTION_ONLY, question, (), ()), disallowed_special=())  # the question alone, never cut
    allowance = min(budget.context_token_budget, budget.context_length - budget.answer_reserve_tokens - budget.safety_margin_tokens - len(base_ids))  # room for evidence
    pairs, passages, passage_texts = select_items(evidence, layout, allowance, tokenizer)  # first selection by the estimate
    while True:  # re-check with the real token count
        prompt_text = render_prompt(layout, question, passage_texts, [(pair.record.question, pair.record.answer) for pair in pairs])  # render
        token_ids = tokenizer.encode(prompt_text, disallowed_special=())  # the real tokens
        fits_context = len(token_ids) + budget.answer_reserve_tokens + budget.safety_margin_tokens <= budget.context_length  # leaves room to answer
        fits_budget = len(token_ids) - len(base_ids) <= budget.context_token_budget  # retrieved text within its budget
        if (fits_context and fits_budget) or not (pairs or passages):  # done, or nothing left to drop
            return PackedPrompt(layout, prompt_text, tuple(token_ids), tuple(passages), tuple(pairs))  # the finished prompt
        if passages:  # the lowest-ranked item is the last passage
            passages.pop()  # drop it
            passage_texts.pop()  # and its text
        else:  # only examples remain
            pairs.pop()  # drop the lowest-ranked example
