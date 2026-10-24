# Game of Thrones answer writer: the fine-tuned GPT-2 medium model continues a prompt token by token with a preallocated cache
import time  # clock for the answer time limit
from dataclasses import dataclass  # frozen result and settings records
from pathlib import Path  # weights file path
from typing import Literal, Sequence  # stop reason names and stop text lists
import tiktoken  # GPT-2 tokenizer
import torch  # tensor library
from gpt2_demo_service.language_models.decoding_cache import DecodingCache  # preallocated keys and values
from gpt2_demo_service.language_models.gpt_model import GPT2_MEDIUM_SIZE, GPTModel, build_config  # network and its size settings
from gpt2_demo_service.language_models.token_sampling import penalise_repeated_tokens, pick_next_token  # next-token choice

END_OF_TEXT_TOKEN_ID = 50256  # training ended every answer with this token, so the model emits it when done
BUILT_IN_STOP_MARKERS = ("###", "Answer the following question")  # the model starts a new block or example with these
StopReason = Literal["end_token", "stop_marker", "token_limit", "time_limit"]  # why an answer stopped


@dataclass(frozen=True)
class SamplingSettings:  # how the next word is chosen
    temperature: float  # randomness
    top_p: float  # share of likely words allowed
    top_k: int  # number of likely words allowed
    repetition_penalty: float  # discourages repeating generated words


@dataclass(frozen=True)
class GeneratedAnswer:  # what one generation returns
    text: str  # the answer without the prompt or any stop marker
    answer_token_count: int  # tokens generated, the end token excluded
    stop_reason: StopReason  # which rule ended it
    prefill_milliseconds: float  # time to read the prompt
    decode_milliseconds: float  # time to write the answer


class PromptTooLongError(ValueError):  # prompt plus answer reserve do not fit the model's context (a packing bug)
    pass


def load_got_text_generator(weights_path: Path, answer_time_limit_seconds: float) -> "GotTextGenerator":  # reads the 1.7 GB weights
    model = GPTModel(build_config(GPT2_MEDIUM_SIZE))  # empty network of the right size
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))  # fill it with the trained weights
    model.eval()  # inference mode, no dropout
    return GotTextGenerator(model, tiktoken.get_encoding("gpt2"), answer_time_limit_seconds)  # ready to write answers


class GotTextGenerator:  # holds the loaded text model and writes answers
    def __init__(self, model: GPTModel, tokenizer: tiktoken.Encoding, answer_time_limit_seconds: float) -> None:  # tests inject a tiny model
        self.model = model  # the network
        self.tokenizer = tokenizer  # turns text into token ids and back
        self.answer_time_limit_seconds = answer_time_limit_seconds  # prefill + decode may not take longer than this
        self.context_length = model.pos_emb.weight.shape[0]  # longest text the model can read

    def create_decoding_cache(self, capacity: int) -> DecodingCache:  # empty cache sized for one request
        first_attention = self.model.trf_blocks[0].att  # attention shape of the first layer (all layers match)
        return DecodingCache(  # one preallocation per request
            len(self.model.trf_blocks), 1, first_attention.head_count, capacity, first_attention.head_dim, self.model.tok_emb.weight.dtype,
        )

    @torch.inference_mode()
    def generate_answer(self, prompt_token_ids: Sequence[int], max_new_tokens: int, sampling: SamplingSettings,  # writes one answer for an already packed prompt
                        stop_strings: Sequence[str], random_generator: torch.Generator | None = None) -> GeneratedAnswer:
        if len(prompt_token_ids) + max_new_tokens > self.context_length:  # the model cannot hold prompt and answer together
            raise PromptTooLongError("%d prompt + %d answer tokens exceed %d" % (len(prompt_token_ids), max_new_tokens, self.context_length))  # a packing bug, never silently cut the prompt
        stop_markers = [*BUILT_IN_STOP_MARKERS, *stop_strings]  # texts that end the answer
        decoding_cache = self.create_decoding_cache(len(prompt_token_ids) + max_new_tokens)  # one allocation for the whole request
        started_at = time.perf_counter()  # clock for prefill and the time limit
        logits = self.model(torch.tensor([list(prompt_token_ids)]), decoding_cache, last_position_only=True)[:, -1, :]  # read the prompt once
        prefill_seconds = time.perf_counter() - started_at  # time spent reading the prompt
        generated_ids: list[int] = []  # the answer tokens only
        stop_reason: StopReason = "token_limit"  # replaced when another rule fires first
        while True:  # one new token per round
            logits = penalise_repeated_tokens(logits, generated_ids, sampling.repetition_penalty)  # discourage loops
            next_token = pick_next_token(logits, sampling.temperature, sampling.top_k, sampling.top_p, random_generator)  # choose with the request settings
            if next_token.item() == END_OF_TEXT_TOKEN_ID:  # the model says the answer is finished
                stop_reason = "end_token"  # natural end
                break  # stop generating
            generated_ids.append(next_token.item())  # keep the token
            if any(marker in self.tokenizer.decode(generated_ids) for marker in stop_markers):  # a next block started
                stop_reason = "stop_marker"  # cut below
                break  # stop generating
            if len(generated_ids) >= max_new_tokens:  # the answer reserve is used up
                break  # stop_reason stays token_limit
            if time.perf_counter() - started_at >= self.answer_time_limit_seconds:  # the caller would give up soon
                stop_reason = "time_limit"  # return what we have
                break  # stop generating
            logits = self.model(next_token, decoding_cache, last_position_only=True)[:, -1, :]  # feed the token back in, one position
        decode_seconds = time.perf_counter() - started_at - prefill_seconds  # everything after the prompt was read
        return GeneratedAnswer(  # the finished answer
            self.cut_at_stop_marker(self.tokenizer.decode(generated_ids), stop_markers), len(generated_ids), stop_reason,
            prefill_seconds * 1000.0, decode_seconds * 1000.0,
        )

    @staticmethod
    def cut_at_stop_marker(answer_text: str, stop_markers: Sequence[str]) -> str:  # drops everything from the first stop marker on
        cut_positions = [answer_text.find(marker) for marker in stop_markers if marker in answer_text]  # where markers start
        return (answer_text[: min(cut_positions)] if cut_positions else answer_text).strip()  # trimmed answer
