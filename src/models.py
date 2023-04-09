import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.key = nn.Linear(args.n_embd, args.head_size, bias=False)
        self.query = nn.Linear(args.n_embd, args.head_size, bias=False)
        self.value = nn.Linear(args.n_embd, args.head_size, bias=False)
        # self.tril = torch.tril(torch.ones((block_size, block_size), requires_grad=False))
        self.register_buffer(
            "tril", torch.tril(torch.ones((args.seq_length, args.seq_length)))
        )

    def forward(self, x):
        b, s, c = x.shape  # (batch, seq_length, channels)
        k = self.key(x)  # (b, s, c) --> (b, s, head_size)
        q = self.query(x)  # (b, s, c) --> (b, s, head_size)

        attn = (
            q @ k.transpose(-2, -1) * self.args.head_size**-0.5
        )  # (b, s, head_size) @ (b, head_size, s) --> (b, s, s)

        attn = attn.masked_fill(self.tril[:s, :s] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        v = self.value(x)  # (b, s, c) --> (b, s, head_size)
        out = attn @ v  # (b, s, s) @ (b, s, head_size) --> (b, s, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        assert args.n_embd % args.n_head == 0

        self.heads = nn.ModuleList([AttentionHead(args) for _ in range(args.n_head)])
        self.proj = nn.Linear(args.n_embd, args.n_embd)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(x)
        return out


class FeedForward(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.n_embd, args.n_embd * 4),
            nn.ReLU(),
            nn.Linear(args.n_embd * 4, args.n_embd),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class Model(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.token_embedding = nn.Embedding(
            num_embeddings=args.vocab_size, embedding_dim=args.n_embd
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=args.seq_length, embedding_dim=args.n_embd
        )
        self.attn = MultiHeadAttention(args)

        self.ffwd = FeedForward(args)

        self.lm_head = nn.Linear(args.n_embd, args.vocab_size)

    def forward(self, idx):
        b, s = idx.shape  # (batch_size, seq_length)
        token_embd = self.token_embedding(idx)  # (b, s, n_embd)
        pos_embd = self.position_embedding(
            torch.arange(s, device=idx.device)
        )  # (b, s, n_embd)
        x = token_embd + pos_embd  # (b, s, n_embd)
        x = self.attn(x)  # (b, s, head_size)
        x = self.ffwd(x)
        logits = self.lm_head(x)
        return logits

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
