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
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.chars = self.model.chars
        self.seq_length = self.model.seq_length
        self.stoi = self.model.stoi
        self.itos = self.model.itos
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda t: "".join([self.itos[i] for i in t])

    @torch.no_grad()
    def predict(self, text: str, max_token=100):
        idx = torch.tensor(self.encode(text), dtype=torch.int64).unsqueeze(
            dim=0
        )  # batch like (b, s, c)
        for _ in range(max_token):
            idx_crop = idx[:, -self.seq_length :]  #
            logits = self.model(idx_crop)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        idx = idx.tolist()[0]
        return self.decode(idx)
