# Makes the grounded training examples of one pair: the oracle passage with hard distractors (or distractors only), shuffled, packed like the Space packs
import random  # seeded coin flip and shuffle
from typing import Mapping, Sequence  # inputs
import tiktoken  # GPT-2 tokenizer
from gpt2_demo_service.index_search.evidence_records import RetrievedEvidence  # what the retriever found for the pair's question
from retrieval_contracts.bundle_records import PassageRecord  # stored passages
from grounding_data_settings import GroundingDataSettings  # numbers
from grounding_prompt_building import encode_grounded_sequence, pack_grounded_prompt  # packing and tokens
from oracle_passage_finding import count_hits_inside  # shingle hits inside a passage


def overlaps(first: PassageRecord, second: PassageRecord) -> bool:  # True when two passages share characters of one part
    return first.part_number == second.part_number and first.character_start < second.character_end and second.character_start < first.character_end  # same part and the character ranges cross


def pick_hard_distractors(evidence: RetrievedEvidence, oracle: PassageRecord, span, count: int, excluded_part: int) -> list[PassageRecord]:  # top retrieved passages that do not hold the answer
    chosen: list[PassageRecord] = []  # distractors so far, best first
    for candidate in evidence.passage_candidates:  # ranked, best first
        record = candidate.record  # the stored passage
        if record.part_number == excluded_part or overlaps(record, oracle) or count_hits_inside(record, span) > 0:  # evaluation text, the oracle itself, or another holder of the answer
            continue  # not a fair distractor
        if any(overlaps(record, kept) for kept in chosen):  # mostly the same text as one already taken
            continue  # next candidate
        chosen.append(record)  # take it
        if len(chosen) == count:  # enough
            break  # done
    return chosen  # may be shorter when retrieval returned too little


def build_pair_example(pair: Mapping, oracle: PassageRecord, span, evidence: RetrievedEvidence, settings: GroundingDataSettings,  # one training sequence for one pair
                       tokenizer: tiktoken.Encoding) -> dict | None:
    shuffler = random.Random(settings.random_seed + int(pair["source_index"]))  # repeatable per pair
    includes_oracle = shuffler.random() < settings.oracle_share  # 80 percent keep the answer passage
    distractor_count = settings.oracle_distractor_count if includes_oracle else settings.distractor_only_count  # how many distractors
    distractors = pick_hard_distractors(evidence, oracle, span, distractor_count, settings.held_out_part_number)  # fair wrong passages
    if len(distractors) < distractor_count:  # retrieval gave too few
        return None  # the pair is skipped
    records: list[PassageRecord] = ([oracle] if includes_oracle else []) + distractors  # what the prompt will show
    shuffler.shuffle(records)  # the answer passage does not always sit in the same place
    packed = pack_grounded_prompt(pair["q"], records, settings, tokenizer)  # the Space's packer
    if includes_oracle and not any(passage.record.passage_id == oracle.passage_id for passage in packed.passages):  # packing dropped the oracle
        return None  # skipped: a training example must really contain its evidence
    example = encode_grounded_sequence(packed, pair["a"], settings, tokenizer)  # tokens and prompt length
    return {**example, "source_index": int(pair["source_index"]), "has_oracle": includes_oracle}  # with its bookkeeping


def build_retrieved_example(question: str, answer: str, evidence: RetrievedEvidence, settings: GroundingDataSettings, tokenizer: tiktoken.Encoding,  # a validation sequence with what retrieval really returns
                            maximum_passages: int) -> dict:
    top_records: Sequence[PassageRecord] = [candidate.record for candidate in evidence.passage_candidates[:maximum_passages]]  # as the Space takes them
    return encode_grounded_sequence(pack_grounded_prompt(question, top_records, settings, tokenizer), answer, settings, tokenizer)  # same path as training
