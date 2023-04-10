import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import CreateDataset, prepare_data
from src.models import Model


def main(args: argparse.Namespace):
    # read config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    dialogues = prepare_data(config["data"]["path"])
    dataset = CreateDataset(
        dialogues,
        seq_length=config["model"]["seq_length"],
        size=config["data"]["train_size"],
    )
    train_ds = dataset.train_dataset()
    val_ds = dataset.test_dataset()
    train_dl = DataLoader(
        dataset=train_ds, batch_size=config["trainer"]["batch_size"], shuffle=True
    )
    val_dl = DataLoader(
        dataset=val_ds, batch_size=config["trainer"]["batch_size"], shuffle=False
    )

    # init model
    config["model"]["vocab_size"] = len(dataset.chars)
    m = Model(**config["model"])
    m.to(device)

    # optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=float(config["optimizer"]["lr"]))

    # wandb
    if args.wandb:
        import wandb

        wandb.init(project=config["wandb"]["project"], config=config)
        wandb.watch(models=m, log_freq=config["wandb"]["log_freq"])  # log every n batch

    # training loop
    epochs = config["trainer"]["max_epoch"]

    for epoch in range(0, epochs):
        # train
        pbar = tqdm(train_dl)
        mean_loss = torch.zeros(1, device=device)  # avg loss for epoch
        running_loss = torch.zeros(1, device=device)
        for ib, (xb, yb) in enumerate(pbar):
            m.train()
            xb, yb = xb.to(device), yb.to(device)

            # feed forward
            logits, loss = m(xb, yb)

            # backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # log
            mean_loss += loss * xb.shape[0]
            running_loss = (ib * running_loss + loss) / (ib + 1)
            pbar.set_description(f"Epoch {epoch}/{epochs}")
            pbar.set_postfix({"train/loss": running_loss.item()})

            if args.wandb and ib % config["wandb"]["log_freq"] == 0:
                wandb.log({"step": epoch * len(train_dl.dataset) + ib})
                wandb.log({"train/loss": loss.item()})

        # validation
        pbar = tqdm(val_dl)
        mean_loss = torch.zeros(1, device=device)  # avg loss for epoch
        running_loss = torch.zeros(1, device=device)  # running mean batch
        for ib, (xb, yb) in enumerate(pbar):
            m.eval()
            xb, yb = xb.to(device), yb.to(device)
            # feed forward
            with torch.no_grad():
                logits, loss = m(xb, yb)
                mean_loss += loss * xb.shape[0]

            # log
            running_loss = (ib * running_loss + loss) / (ib + 1)
            pbar.set_postfix({"validation/loss": running_loss.item()})
        mean_loss /= len(val_dl.dataset)
        pbar.set_postfix({"validation/loss": mean_loss.item()})
        if args.wandb:
            wandb.log({"epoch": epoch, "validation/loss": mean_loss.item()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    main(args)
