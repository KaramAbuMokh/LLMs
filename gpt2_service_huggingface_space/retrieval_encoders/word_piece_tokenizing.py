# Turns text into the padded integer inputs a pinned encoder graph declares (tokenizer loading, truncation, per-batch padding)
from pathlib import Path  # model folder path
from typing import Sequence  # encodings in, names in
import numpy  # padded input arrays
from tokenizers import Encoding, Tokenizer  # word-piece tokenizer
from retrieval_contracts.encoder_contracts import PinnedModelFiles  # which tokenizer file to read

PADDING_TOKEN_ID = 0  # [PAD] in both pinned vocabularies


def build_word_piece_tokenizer(model_folder: Path, files: PinnedModelFiles, input_token_limit: int) -> Tokenizer:  # tokenizer that cuts at the limit
    if not 0 < input_token_limit <= files.max_input_tokens:  # the graph cannot read more than its position table holds
        raise ValueError("input_token_limit must be between 1 and %d" % files.max_input_tokens)  # refuse at load time
    tokenizer = Tokenizer.from_file(str(model_folder / files.tokenizer_file))  # the pinned tokenizer JSON
    tokenizer.enable_truncation(max_length=input_token_limit)  # longer inputs are cut (the longest side of a pair first)
    return tokenizer  # ready to encode single texts and (question, passage) pairs


def pad_encodings_into_inputs(encodings: Sequence[Encoding], declared_input_names: Sequence[str]) -> dict[str, numpy.ndarray]:  # one batch for the graph
    batch_width = max(len(encoding.ids) for encoding in encodings)  # pad to the longest text of this batch only
    input_ids = numpy.full((len(encodings), batch_width), PADDING_TOKEN_ID, dtype=numpy.int64)  # token ids, padding by default
    attention_mask = numpy.zeros((len(encodings), batch_width), dtype=numpy.int64)  # 1 for real tokens
    token_type_ids = numpy.zeros((len(encodings), batch_width), dtype=numpy.int64)  # 0 for the first text, 1 for the second of a pair
    for row, encoding in enumerate(encodings):  # fill one row per text
        token_count = len(encoding.ids)  # real tokens of this text
        input_ids[row, :token_count] = encoding.ids  # token ids
        attention_mask[row, :token_count] = 1  # mark them real
        token_type_ids[row, :token_count] = encoding.type_ids  # segment ids
    available_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}  # everything we can supply
    return {name: available_inputs[name] for name in declared_input_names}  # only what the graph declares (a KeyError means a new graph kind)
