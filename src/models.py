import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, seq_length, dropout: float = 0.0) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # self.tril = torch.tril(torch.ones((block_size, block_size), requires_grad=False))
        self.register_buffer("tril", torch.tril(torch.ones((seq_length, seq_length))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, s, c = x.shape  # (batch, seq_length, channels)
        k = self.key(x)  # (b, s, c) --> (b, s, head_size)
        q = self.query(x)  # (b, s, c) --> (b, s, head_size)

        attn = (
            q @ k.transpose(-2, -1) * k.size(-1) ** -0.5
        )  # (b, s, head_size) @ (b, head_size, s) --> (b, s, s)

        attn = attn.masked_fill(self.tril[:s, :s] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        v = self.value(x)  # (b, s, c) --> (b, s, head_size)
        out = attn @ v  # (b, s, s) @ (b, s, head_size) --> (b, s, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_size, seq_length, dropout) -> None:
        super().__init__()
        assert n_embd % n_head == 0

        self.heads = nn.ModuleList(
            [
                AttentionHead(n_embd, head_size, seq_length, dropout)
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, head_size, seq_length, dropout) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(n_embd, n_head, head_size, seq_length, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd,
        n_head,
        n_layer,
        head_size,
        seq_length,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=n_embd
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=seq_length, embedding_dim=n_embd
        )
        self.decoder_block = nn.Sequential(
            *(
                DecoderBlock(n_embd, n_head, head_size, seq_length, dropout)
                for _ in range(n_layer)
            )
        )

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, target=None):
        b, s = idx.shape  # (batch_size, seq_length)
        token_embd = self.token_embedding(idx)  # (b, s, n_embd)
        pos_embd = self.position_embedding(
            torch.arange(s, device=idx.device)
        )  # (b, s, n_embd)
        x = token_embd + pos_embd  # (b, s, n_embd)
        x = self.decoder_block(x)
        logits = self.lm_head(x)
        if target is None:
            loss = None
        else:
            b, s, c = logits.shape
            logits = logits.view(b * s, -1)
            loss = F.cross_entropy(logits, target.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_tokens):
        # idx (b, s)
        for _ in range(max_tokens):
            idx_crop = idx[:, -self.args.seq_length :]

            logits = self(idx_crop)  # (b, s, vocab_size)
            logits = logits[:, -1, :]  # (b, 1, vocab_size)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
