# Tests that grounded sequences are packed exactly like the Space packs them and masked correctly
import tiktoken  # GPT-2 tokenizer
from grounding_data_settings import GroundingDataSettings  # numbers
from grounding_prompt_building import END_OF_TEXT_TOKEN_ID, encode_grounded_sequence, pack_grounded_prompt  # code under test
from retrieval_contracts.bundle_records import PassageRecord  # stored passage record


def make_passage(passage_id: str, start: int) -> PassageRecord:  # a passage with invented text, far from the others
    return PassageRecord(passage_id, 0, "Book", 1, 1, "Title", start, start + 400, 30, "An invented sentence about a lamp. Another invented sentence about a key %d." % start)  # invented text only


def test_prompt_ends_with_the_trained_template_and_answer_follows() -> None:  # prompt = passages + template, then answer + end token
    tokenizer, settings = tiktoken.get_encoding("gpt2"), GroundingDataSettings()  # real tokenizer, production numbers
    packed = pack_grounded_prompt("Who holds the key?", [make_passage("a", 0), make_passage("b", 1000)], settings, tokenizer)  # two invented passages
    example = encode_grounded_sequence(packed, "The lamp keeper holds it.", settings, tokenizer)  # the training sequence
    assert packed.prompt_text.endswith("### question:\nWho holds the key?\n\n### Response:\n")  # the exact trained template last
    assert example["tokens"][-1] == END_OF_TEXT_TOKEN_ID and example["tokens"].count(END_OF_TEXT_TOKEN_ID) == 1  # one end token, at the end
    assert example["tokens"][:example["prompt_length"]] == list(packed.prompt_token_ids)  # the unscored part is the packed prompt
    assert example["passage_count"] == 2 and "invented sentence" in packed.prompt_text  # both passages reached the prompt


def test_answer_is_cut_to_the_maximum_length() -> None:  # an over-long answer is cut like the Q&A training data
    tokenizer, settings = tiktoken.get_encoding("gpt2"), GroundingDataSettings()  # real tokenizer, production numbers
    packed = pack_grounded_prompt("Who holds the key?", [make_passage("a", 0)], settings, tokenizer)  # one passage
    example = encode_grounded_sequence(packed, "word " * 400, settings, tokenizer)  # a very long answer
    assert len(example["tokens"]) - example["prompt_length"] == settings.maximum_answer_tokens + 1  # answer tokens plus the end token
