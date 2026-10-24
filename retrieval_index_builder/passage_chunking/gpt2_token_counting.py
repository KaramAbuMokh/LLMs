# Counts GPT-2 tokens, the unit of the model's 1,024-token window and of every passage size
import tiktoken  # the model's own tokenizer

_GPT2_ENCODING = tiktoken.get_encoding("gpt2")  # loaded once; the vocabulary downloads on first use (cached afterwards)


def count_gpt2_tokens(text: str) -> int:  # tokens the model sees for this text (special-token text counts as plain text, as in training)
    return len(_GPT2_ENCODING.encode_ordinary(text))  # same ids as encode(..., disallowed_special=())
