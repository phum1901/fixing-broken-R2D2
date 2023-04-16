from pathlib import Path

import torch


class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.best_score = float("inf")
        self.counter = 0
        self.min_delta = min_delta

    def __call__(self, value):
        stop = False
        if value - self.min_delta < self.best_score:
            self.best_score = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                stop = True
        return stop


class ModelCheckpoint:
    def __init__(self, save_path: str, mode="min", save_best_only=True):
        self.save_path = Path(save_path)
        self.mode = mode
        self.save_best_only = save_best_only  # TODO
        self.best_score = None

        self.save_path.mkdir(parents=True, exist_ok=True)

    def __call__(self, value, model, optimizer, epoch, config):
        if self.mode == "min":
            score = value
        else:
            score = -value

        if self.best_score is None or score < self.best_score:
            self.best_score = score
            m = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": value,
                "config": config,
            }

            # if self.save_best_only:
            #     save_path = self.save_path / f"best_model.ckpt"
            #     torch.save(
            #         m,
            #         save_path,
            #     )
            #     return m
            # else:
            save_path = self.save_path / f"epoch={epoch}&validation.loss={value:.4f}.ckpt"
            torch.save(
                m,
                save_path,
            )
            return str(save_path)
