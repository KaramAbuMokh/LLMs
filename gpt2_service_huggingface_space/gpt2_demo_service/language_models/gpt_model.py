# The GPT-2 network (small or medium) shared by the Game of Thrones model and the spam classifier
import torch  # tensor library
import torch.nn as nn  # neural network layers
from gpt2_demo_service.language_models.decoding_cache import DecodingCache  # cache of keys and values for incremental decoding
from gpt2_demo_service.language_models.gpt_blocks import LayerNorm, TransformerBlock  # building blocks

BASE_CONFIG = {  # settings both models share
    "vocab_size": 50257,  # number of tokens GPT-2 knows
    "context_length": 1024,  # longest text the model can read
    "dropout_rate": 0.05,  # dropout used during training
    "query_key_value_bias": True,  # attention projections have a bias
}
GPT2_SMALL_SIZE = {"embedding_dim": 768, "layer_count": 12, "head_count": 12}  # 124M, spam classifier
GPT2_MEDIUM_SIZE = {"embedding_dim": 1024, "layer_count": 24, "head_count": 16}  # 355M, Game of Thrones


def build_config(model_size):  # merge the shared settings with one size, without changing either
    return {**BASE_CONFIG, **model_size}  # fresh dictionary for one model


class GPTModel(nn.Module):  # token ids in, one score per vocabulary word out
    def __init__(self, config):  # config comes from build_config
        super().__init__()  # register this module with PyTorch
        self.tok_emb = nn.Embedding(config["vocab_size"], config["embedding_dim"])  # word vectors (name fixed by weights)
        self.pos_emb = nn.Embedding(config["context_length"], config["embedding_dim"])  # position vectors (name fixed by weights)
        self.drop_emb = nn.Dropout(config["dropout_rate"])  # switched off by eval mode
        self.trf_blocks = nn.Sequential(  # stack of transformer blocks (name fixed by weights)
            *[TransformerBlock(config) for _ in range(config["layer_count"])]
        )
        self.final_norm = LayerNorm(config["embedding_dim"])  # last normalisation
        self.out_head = nn.Linear(config["embedding_dim"], config["vocab_size"], bias=False)  # scores per word

    def forward(self, token_ids, decoding_cache: DecodingCache | None = None, last_position_only: bool = False):
        # token_ids has shape (batch, new tokens); with a cache they continue after the cached positions
        past_count = 0 if decoding_cache is None else decoding_cache.cached_position_count  # position of the first new token
        positions = torch.arange(past_count, past_count + token_ids.shape[1], device=token_ids.device)  # absolute positions
        x = self.drop_emb(self.tok_emb(token_ids) + self.pos_emb(positions))  # word plus position vectors
        for layer_number, block in enumerate(self.trf_blocks):  # every transformer block in order
            x = block(x, None if decoding_cache is None else decoding_cache.layers[layer_number])  # its own layer cache
        x = self.final_norm(x)  # last normalisation
        if last_position_only:  # decoding needs the next-word scores of the final position only
            x = x[:, -1:, :]  # skip the output head for every earlier position
        return self.out_head(x)  # scores for the next word at every kept position
