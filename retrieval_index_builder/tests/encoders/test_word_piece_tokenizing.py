# Tokenizer loading, truncation and batch padding, tested with a tiny invented vocabulary (no model files needed)
import numpy  # array checks
import pytest  # expected errors
from tokenizers import Tokenizer, models, pre_tokenizers, processors  # build a small tokenizer
from retrieval_contracts.encoder_contracts import PinnedModelFiles  # model description the code reads
from retrieval_encoders.word_piece_tokenizing import PADDING_TOKEN_ID, build_word_piece_tokenizer, pad_encodings_into_inputs  # code under test

TINY_FILES = PinnedModelFiles("fixture/tiny", "0" * 40, "onnx/model.onnx", "tokenizer.json", 16)  # a model that reads at most 16 word pieces
WORDS = "who forged the blade mira smith kettle ford".split()  # vocabulary words


def write_tiny_tokenizer(model_folder) -> None:  # a word-level tokenizer with BERT-style special tokens and pair support
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, **{word: number + 4 for number, word in enumerate(WORDS)}}  # token ids
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))  # word-level model
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # split on spaces and punctuation
    tokenizer.post_processor = processors.TemplateProcessing(  # [CLS] a [SEP] and [CLS] a [SEP] b [SEP]
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 2), ("[SEP]", 3)],  # how single texts and pairs get their special tokens
    )
    tokenizer.save(str(model_folder / "tokenizer.json"))  # the file the loader reads


def test_limit_must_be_between_one_and_the_models_maximum(tmp_path):  # a limit the graph cannot read is refused at load time
    write_tiny_tokenizer(tmp_path)  # tokenizer file
    for bad_limit in (0, -1, 17):  # none, negative, above the model's 16
        with pytest.raises(ValueError):  # must refuse
            build_word_piece_tokenizer(tmp_path, TINY_FILES, bad_limit)  # code under test


def test_long_text_is_cut_to_the_limit_special_tokens_included(tmp_path):  # truncation of a single text
    write_tiny_tokenizer(tmp_path)  # tokenizer file
    tokenizer = build_word_piece_tokenizer(tmp_path, TINY_FILES, 8)  # limit 8
    encoding = tokenizer.encode(" ".join(WORDS * 5))  # 40 words
    assert len(encoding.ids) == 8 and encoding.ids[0] == 2 and encoding.ids[-1] == 3  # [CLS] ... [SEP] still frame the text


def test_pair_is_cut_on_its_longer_side_and_keeps_both_segments(tmp_path):  # question short, passage long
    write_tiny_tokenizer(tmp_path)  # tokenizer file
    tokenizer = build_word_piece_tokenizer(tmp_path, TINY_FILES, 10)  # limit 10
    encoding = tokenizer.encode("who forged", " ".join(WORDS * 5))  # a 2-word question and a 40-word passage
    assert len(encoding.ids) == 10  # at the limit
    assert encoding.type_ids[:4] == [0, 0, 0, 0] and set(encoding.type_ids) == {0, 1}  # question segment first, passage segment second
    assert encoding.ids[1:3] == [4, 5]  # the whole question survives


def test_padding_is_per_batch_and_only_declared_inputs_are_returned(tmp_path):  # shapes, masks and input selection
    write_tiny_tokenizer(tmp_path)  # tokenizer file
    tokenizer = build_word_piece_tokenizer(tmp_path, TINY_FILES, 16)  # limit 16
    encodings = tokenizer.encode_batch(["who forged", "who forged the blade mira"])  # 4 and 7 tokens with special tokens
    graph_inputs = pad_encodings_into_inputs(encodings, ["input_ids", "attention_mask"])  # code under test
    assert sorted(graph_inputs) == ["attention_mask", "input_ids"]  # token_type_ids not declared, so not returned
    assert graph_inputs["input_ids"].shape == (2, 7) and graph_inputs["input_ids"].dtype == numpy.int64  # padded to the longest text
    assert list(graph_inputs["attention_mask"][0]) == [1, 1, 1, 1, 0, 0, 0] and list(graph_inputs["input_ids"][0][4:]) == [PADDING_TOKEN_ID] * 3  # padding rows
    with pytest.raises(KeyError):  # a graph asking for an input we cannot build is a new kind of model
        pad_encodings_into_inputs(encodings, ["position_ids"])  # code under test
