# Builds grounded training and validation sequences with the SERVICE's own packer, so training and serving prompts cannot drift
from typing import Sequence  # inputs
import tiktoken  # GPT-2 tokenizer
from gpt2_demo_service.grounded_answers.answer_settings import PromptLayout  # layout names
from gpt2_demo_service.grounded_answers.context_packing import PackedPrompt, PromptBudget, pack_prompt  # the packer
from gpt2_demo_service.index_search.evidence_records import EvidenceRoute, RetrievedEvidence, ScoredPassage  # evidence records
from retrieval_contracts.bundle_records import PassageRecord  # stored passages
from grounding_data_settings import GroundingDataSettings  # numbers (sibling module: the build scripts run from this folder)

END_OF_TEXT_TOKEN_ID = 50256  # GPT-2 end token that closes every answer
CONTEXT_LENGTH = 1024  # the model's window


def wrap_passages_as_evidence(records: Sequence[PassageRecord]) -> RetrievedEvidence:  # passages in the order the prompt should show them (best first)
    scored = tuple(ScoredPassage(record, rank, 0.0, 0.0, None) for rank, record in enumerate(records, start=1))  # scores do not matter to the packer
    return RetrievedEvidence(EvidenceRoute.PASSAGES_ONLY, scored, scored, (), (), None, {})  # no pair examples, no timings


def pack_grounded_prompt(question: str, records: Sequence[PassageRecord], settings: GroundingDataSettings, tokenizer: tiktoken.Encoding) -> PackedPrompt:  # the prompt the Space would build
    budget = PromptBudget(CONTEXT_LENGTH, settings.answer_reserve_tokens, settings.context_token_budget, settings.safety_margin_tokens)  # as the service
    return pack_prompt(question, wrap_passages_as_evidence(records), PromptLayout.PASSAGES_BEFORE_TEMPLATE, budget, tokenizer)  # same call as at serving time


def encode_grounded_sequence(packed: PackedPrompt, answer: str, settings: GroundingDataSettings, tokenizer: tiktoken.Encoding) -> dict:  # tokens and the length of the unscored prompt
    answer_ids = tokenizer.encode_ordinary(answer)[:settings.maximum_answer_tokens]  # the answer, cut like the Q&A data
    return {"tokens": list(packed.prompt_token_ids) + answer_ids + [END_OF_TEXT_TOKEN_ID], "prompt_length": len(packed.prompt_token_ids),  # end token always last
            "passage_count": len(packed.passages)}  # how many passages survived packing
