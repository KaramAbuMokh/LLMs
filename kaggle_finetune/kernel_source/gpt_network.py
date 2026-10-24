# The GPT-2 network with the SAME parameter names as Karam's saved weights, using PyTorch's memory-saving attention for training
import torch  # tensor library
import torch.nn as nn  # neural network layers
import torch.nn.functional as functional  # fused layer norm and scaled dot-product attention
from torch.utils.checkpoint import checkpoint  # recompute block activations in backward to save GPU memory

VOCABULARY_SIZE = 50257  # GPT-2 tokens
CONTEXT_LENGTH = 1024  # longest text the model reads
END_OF_TEXT_TOKEN_ID = 50256  # "<|endoftext|>", also used as padding


class LayerNorm(nn.Module):  # normalises each token vector, then a learned scale and shift (names fixed by the weights)
    def __init__(self, embedding_dim):  # embedding_dim is the width of one token vector
        super().__init__()  # register with PyTorch
        self.eps = 1e-5  # avoids dividing by zero
        self.scale = nn.Parameter(torch.ones(embedding_dim))  # learned multiplier
        self.shift = nn.Parameter(torch.zeros(embedding_dim))  # learned offset

    def forward(self, x):  # same maths as Karam's hand-written version (biased variance, eps inside the root), fused kernel
        return functional.layer_norm(x, (x.shape[-1],), self.scale, self.shift, self.eps)  # far less memory than the manual steps


class MultiHeadAttention(nn.Module):  # causal self-attention; same weights as the original, faster kernel
    def __init__(self, embedding_dim, head_count, dropout_rate):  # sizes of one attention layer
        super().__init__()  # register with PyTorch
        self.head_count = head_count  # number of heads
        self.head_dim = embedding_dim // head_count  # width per head
        self.dropout_rate = 0.0  # no dropout on attention weights: keeps the T4 on the memory-saving kernel (block dropout stays on)
        self.W_query = nn.Linear(embedding_dim, embedding_dim, bias=True)  # name fixed by the weights
        self.W_key = nn.Linear(embedding_dim, embedding_dim, bias=True)  # name fixed by the weights
        self.W_value = nn.Linear(embedding_dim, embedding_dim, bias=True)  # name fixed by the weights
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)  # mixes the heads back together
        self.register_buffer("mask", torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1))  # kept so saved files match

    def forward(self, x):  # x is (batch, tokens, embedding_dim)
        batch_size, token_count, embedding_dim = x.shape  # sizes for reshaping
        split_shape = (batch_size, token_count, self.head_count, self.head_dim)  # one slice per head
        queries = self.W_query(x).view(*split_shape).transpose(1, 2)  # (batch, heads, tokens, head_dim)
        keys = self.W_key(x).view(*split_shape).transpose(1, 2)  # same shape
        values = self.W_value(x).view(*split_shape).transpose(1, 2)  # same shape
        dropout_probability = self.dropout_rate if self.training else 0.0  # no dropout when evaluating
        context = functional.scaled_dot_product_attention(  # softmax(QK^T / sqrt(d)) V with a causal mask
            queries, keys, values, dropout_p=dropout_probability, is_causal=True,
        )
        context = context.transpose(1, 2).contiguous().view(batch_size, token_count, embedding_dim)  # join heads
        return self.out_proj(context)  # final mixing layer


class TransformerBlock(nn.Module):  # attention then feed-forward, each with a shortcut
    def __init__(self, embedding_dim, head_count, dropout_rate):  # sizes of one block
        super().__init__()  # register with PyTorch
        self.att = MultiHeadAttention(embedding_dim, head_count, dropout_rate)  # name fixed by the weights
        self.ff = nn.Module()  # holder so the weight names stay "ff.layer.0" and "ff.layer.2"
        self.ff.layer = nn.Sequential(  # expand, GELU, shrink
            nn.Linear(embedding_dim, 4 * embedding_dim), nn.GELU(approximate="tanh"), nn.Linear(4 * embedding_dim, embedding_dim),
        )
        self.norm1 = LayerNorm(embedding_dim)  # before attention
        self.norm2 = LayerNorm(embedding_dim)  # before feed-forward
        self.drop_shortcut = nn.Dropout(dropout_rate)  # dropout on each branch

    def forward(self, x):  # x is (batch, tokens, embedding_dim)
        x = x + self.drop_shortcut(self.att(self.norm1(x)))  # attention with shortcut
        return x + self.drop_shortcut(self.ff.layer(self.norm2(x)))  # feed-forward with shortcut


class GPTModel(nn.Module):  # token ids in, next-token scores out
    def __init__(self, model_size, dropout_rate):  # model_size has embedding_dim, layer_count, head_count
        super().__init__()  # register with PyTorch
        embedding_dim = model_size["embedding_dim"]  # token vector width
        self.tok_emb = nn.Embedding(VOCABULARY_SIZE, embedding_dim)  # word vectors
        self.pos_emb = nn.Embedding(CONTEXT_LENGTH, embedding_dim)  # position vectors
        self.drop_emb = nn.Dropout(dropout_rate)  # dropout on the input vectors
        self.trf_blocks = nn.Sequential(  # the transformer stack
            *[TransformerBlock(embedding_dim, model_size["head_count"], dropout_rate) for _ in range(model_size["layer_count"])]
        )
        self.final_norm = LayerNorm(embedding_dim)  # last normalisation
        self.out_head = nn.Linear(embedding_dim, VOCABULARY_SIZE, bias=False)  # scores per word (not tied to tok_emb)
        self.use_gradient_checkpointing = False  # set by load_model from the settings

    def hidden_states(self, token_ids):  # last-layer vectors before the word scores (training computes scores in chunks)
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)  # 0, 1, 2, ...
        x = self.drop_emb(self.tok_emb(token_ids) + self.pos_emb(positions))  # word plus position vectors
        for block in self.trf_blocks:  # every transformer block
            if self.use_gradient_checkpointing and self.training:  # keep only block inputs, recompute the rest in backward
                x = checkpoint(block, x, use_reentrant=False)  # about 30% more compute, a fraction of the memory
            else:  # evaluation or checkpointing off
                x = block(x)  # plain forward
        return self.final_norm(x)  # normalised vectors

    def forward(self, token_ids):  # token_ids is (batch, tokens)
        return self.out_head(self.hidden_states(token_ids))  # next-token scores at every position


def load_model(settings, checkpoint_path, device):  # builds the network and fills it with the current weights
    model = GPTModel(settings.model_size, settings.dropout_rate)  # empty network
    model.use_gradient_checkpointing = settings.gradient_checkpointing  # memory switch
    if settings.load_current_checkpoint:  # real runs continue from Karam's fine-tuned weights
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)  # plain tensors only
        model.load_state_dict(state, strict=True)  # every name and shape must match
    return model.to(device)  # on the GPU (or CPU in dry mode)
