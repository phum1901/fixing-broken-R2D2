import torch
import torch.nn.functional as F

from src.models import Model


def save_model_checkpoint_to_torchscript(ckpt_path: str, save_path: str):
    ckpt = torch.load(ckpt_path)
    model = Model(**ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    scripted_model = torch.jit.script(model)
    scripted_model.save(save_path)


class StageModel:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = "model.pt"
        self.model = torch.jit.load(model_path)
        self.chars = self.model.chars
        self.seq_length = self.model.seq_length
        self.stoi = self.model.stoi
        self.itos = self.model.itos
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda t: "".join([self.itos[i] for i in t])

    @torch.no_grad()
    def generate(self, text, max_tokens=100, temperature=1.0, top_k=None):
        idx = torch.tensor(self.encode(text), dtype=torch.int64).unsqueeze(
            dim=0
        )  # batch like (b, s, c)
        for _ in range(max_tokens):
            idx_crop = idx[:, -self.seq_length :]

            logits = self.model(idx_crop)  # (b, s, vocab_size)
            logits = logits[:, -1, :] / temperature  # (b, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < torch.min(v)] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        idx = idx.tolist()[0]
        return self.decode(idx)
