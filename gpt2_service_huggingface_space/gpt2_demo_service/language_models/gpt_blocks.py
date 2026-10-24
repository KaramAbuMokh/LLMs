# Transformer building blocks of the two fine-tuned GPT-2 models; attribute names must match the saved weights
import torch  # tensor library
import torch.nn as nn  # neural network layers
from gpt2_demo_service.language_models.decoding_cache import LayerKeyValueCache  # per-layer cache of keys and values


class LayerNorm(nn.Module):  # normalises each token vector, then applies a learned scale and shift
    def __init__(self, embedding_dim):  # embedding_dim is the width of one token vector
        super().__init__()  # register this module with PyTorch
        self.eps = 1e-5  # tiny number that avoids dividing by zero
        self.scale = nn.Parameter(torch.ones(embedding_dim))  # learned multiplier
        self.shift = nn.Parameter(torch.zeros(embedding_dim))  # learned offset

    def forward(self, x):  # normalise over the last dimension
        mean = x.mean(dim=-1, keepdim=True)  # average value per token
        variance = x.var(dim=-1, keepdim=True, unbiased=False)  # spread per token
        normalised = (x - mean) / torch.sqrt(variance + self.eps)  # zero mean, unit spread
        return self.scale * normalised + self.shift  # apply learned scale and shift


class FeedForward(nn.Module):  # two linear layers with a GELU in between
    def __init__(self, embedding_dim):  # embedding_dim is the width of one token vector
        super().__init__()  # register this module with PyTorch
        self.layer = nn.Sequential(  # name "layer" is fixed by the saved weights
            nn.Linear(embedding_dim, 4 * embedding_dim),  # expand to four times the width
            nn.GELU(approximate="tanh"),  # same tanh GELU the original code wrote by hand
            nn.Linear(4 * embedding_dim, embedding_dim),  # shrink back to the token width
        )

    def forward(self, x):  # run the token vectors through the two layers
        return self.layer(x)  # transformed token vectors


class MultiHeadAttention(nn.Module):  # causal self-attention split over several heads
    def __init__(self, embedding_dim, context_length, dropout_rate, head_count, query_key_value_bias):  # sizes and the dropout rate of one attention layer
        super().__init__()  # register this module with PyTorch
        self.head_count = head_count  # how many attention heads
        self.head_dim = embedding_dim // head_count  # width of each head
        self.W_query = nn.Linear(embedding_dim, embedding_dim, bias=query_key_value_bias)  # name fixed by weights
        self.W_key = nn.Linear(embedding_dim, embedding_dim, bias=query_key_value_bias)  # name fixed by weights
        self.W_value = nn.Linear(embedding_dim, embedding_dim, bias=query_key_value_bias)  # name fixed by weights
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)  # mixes the heads back together
        self.dropout = nn.Dropout(dropout_rate)  # switched off by eval mode
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))  # hides future tokens

    def forward(self, x, layer_cache: LayerKeyValueCache | None = None):  # x has shape (batch, new tokens, embedding_dim)
        batch_size, token_count, embedding_dim = x.shape  # sizes needed for reshaping
        split_shape = (batch_size, token_count, self.head_count, self.head_dim)  # one slice per head
        keys = self.W_key(x).view(*split_shape).transpose(1, 2)  # keys of the new tokens per head
        queries = self.W_query(x).view(*split_shape).transpose(1, 2)  # queries of the new tokens per head
        values = self.W_value(x).view(*split_shape).transpose(1, 2)  # values of the new tokens per head
        past_count = 0 if layer_cache is None else layer_cache.filled_count  # tokens already cached before these
        if layer_cache is not None:  # incremental decoding
            keys, values = layer_cache.append(keys, values)  # new tokens join the cached ones (views, no copy)
        scores = queries @ keys.transpose(2, 3)  # how much each new token looks at every cached and new token
        future_tokens = self.mask.bool()[past_count:past_count + token_count, :past_count + token_count]  # positions a token may not see
        scores = scores.masked_fill(future_tokens, -torch.inf)  # block looking ahead
        weights = torch.softmax(scores / keys.shape[-1] ** 0.5, dim=-1)  # attention weights per token
        weights = self.dropout(weights)  # no-op in eval mode
        context = (weights @ values).transpose(1, 2)  # weighted values, heads next to each other
        context = context.contiguous().view(batch_size, token_count, embedding_dim)  # join the heads
        return self.out_proj(context)  # final mixing layer


class TransformerBlock(nn.Module):  # attention then feed-forward, each with a shortcut connection
    def __init__(self, config):  # config holds the model size settings
        super().__init__()  # register this module with PyTorch
        self.att = MultiHeadAttention(  # attention part (name fixed by weights)
            config["embedding_dim"], config["context_length"], config["dropout_rate"],
            config["head_count"], config["query_key_value_bias"],
        )
        self.ff = FeedForward(config["embedding_dim"])  # feed-forward part (name fixed by weights)
        self.norm1 = LayerNorm(config["embedding_dim"])  # normalise before attention
        self.norm2 = LayerNorm(config["embedding_dim"])  # normalise before feed-forward
        self.drop_shortcut = nn.Dropout(config["dropout_rate"])  # switched off by eval mode

    def forward(self, x, layer_cache: LayerKeyValueCache | None = None):  # x has shape (batch, new tokens, embedding_dim)
        x = x + self.drop_shortcut(self.att(self.norm1(x), layer_cache))  # attention with shortcut
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))  # feed-forward with shortcut
        return x  # updated token vectors
