# What the index builder, the service and the evaluation must agree on about the two small encoders:
# the pinned public model files, the fingerprint stored in the bundle, and the calling interface (implementations live in retrieval_encoders/)
from dataclasses import dataclass  # frozen specification rows
from typing import TYPE_CHECKING, Protocol, Sequence  # structural interfaces

if TYPE_CHECKING:  # numpy is named only for type checkers; this package itself needs nothing beyond the standard library
    import numpy  # array element types
    from numpy.typing import NDArray  # array type hints


@dataclass(frozen=True)
class PinnedModelFiles:  # one public model on the Hugging Face Hub, pinned to an exact commit
    model_id: str  # Hub repo id
    revision: str  # full 40-character commit hash (never a branch name)
    model_file: str  # ONNX graph path inside the repo
    tokenizer_file: str  # tokenizers JSON path inside the repo
    max_input_tokens: int  # longer inputs are cut to this many word pieces


TEXT_ENCODER_FILES = PinnedModelFiles(  # dense encoder for passages and questions (MIT licence, 33M parameters, 384 dimensions)
    "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", "onnx/model.onnx", "tokenizer.json", 512,
)
PASSAGE_PAIR_SCORER_FILES = PinnedModelFiles(  # optional cross-encoder reranker (Apache-2.0, 22M parameters)
    "cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a", "onnx/model.onnx", "tokenizer.json", 512,
)
TEXT_ENCODER_DIMENSIONS = 384  # width of every bge-small vector
TEXT_ENCODER_POOLING = "cls"  # bge takes the first token's output vector, then scales it to unit length
PASSAGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "  # bge v1.5 prefix for short questions against long passages
# The instruction goes ONLY before a visitor question when searching book passages; never before stored texts, never for question-to-question matching


@dataclass(frozen=True)
class EncoderFingerprint:  # identifies exactly which encoder made a set of vectors; the manifest stores one, a loaded encoder reports one
    model_id: str  # Hub repo id (or a fixture name)
    revision: str  # commit hash the model files came from
    model_file_sha256: str  # checksum of the ONNX file actually run
    output_dimensions: int  # vector width
    pooling: str  # how token vectors become one vector ("cls" for bge)
    passage_query_instruction: str  # prefix for passage-search questions (part of the fingerprint: calibrated thresholds depend on it)


class TextEncoder(Protocol):  # turns texts into unit-length vectors; the same object kind runs offline (index) and online (questions)
    @property
    def fingerprint(self) -> EncoderFingerprint:  # which model and files this encoder runs
        ...

    def encode_texts(self, texts: Sequence[str]) -> "NDArray[numpy.float32]":  # (len(texts), dimensions), every row unit length, no prefix added
        ...


class PassagePairScorer(Protocol):  # cross-encoder: reads a question and a passage together and scores how well they fit
    def score_pairs(self, question: str, passage_texts: Sequence[str]) -> "NDArray[numpy.float32]":  # one raw score per passage, higher fits better
        ...
