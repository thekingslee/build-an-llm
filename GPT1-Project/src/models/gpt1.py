import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, seq_len, _ = x.shape

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.reshape(N, seq_len, self.heads, self.head_dim).transpose(1, 2)
        keys = keys.reshape(N, seq_len, self.heads, self.head_dim).transpose(1, 2)
        queries = queries.reshape(N, seq_len, self.heads, self.head_dim).transpose(1, 2)

        energy = torch.matmul(queries, keys.transpose(-2, -1))
        energy = energy / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        energy = energy.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, values)

        out = out.transpose(1, 2).contiguous().reshape(N, seq_len, self.embed_size)
        return self.fc_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size)
        )

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ff(x))
        return x


class GPT1(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, config.EMBED_SIZE)
        self.position_embedding = nn.Embedding(config.MAX_LEN, config.EMBED_SIZE)

        self.layers = nn.ModuleList(
            [TransformerBlock(config.EMBED_SIZE, config.HEADS) for _ in range(config.NUM_LAYERS)]
        )

        self.norm = nn.LayerNorm(config.EMBED_SIZE)
        self.fc_out = nn.Linear(config.EMBED_SIZE, vocab_size)

    def forward(self, x):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(x.device)

        x = self.token_embedding(x) + self.position_embedding(positions)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.fc_out(x)
