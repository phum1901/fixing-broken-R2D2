import argparse
import os
from pathlib import Path

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
    # wandb
    if args.wandb:
        import wandb

        try:
            wandb.init(project=config["wandb"]["project"], config=config)
            config = wandb.config
        except:
            wandb.init(config=config)
            config = wandb.config
        # wandb.watch(models=m, log_freq=config["wandb"]["log_freq"])  # log every n batch
    # for reproducibility
    torch.manual_seed(config["seed"])
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
    config["model"]["chars"] = dataset.chars
    m = Model(**config["model"])
    m.to(device)
    if args.wandb:
        wandb.watch(models=m, log_freq=config["wandb"]["log_freq"])  # log every n batch

    # # wandb
    # if args.wandb:
    #     import wandb

    #     wandb.init(project=config["wandb"]["project"], config=config)
    #     wandb.watch(models=m, log_freq=config["wandb"]["log_freq"])  # log every n batch
    # optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=float(config["optimizer"]["lr"]))

    # poor man's scheduler (will be fix later)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["scheduler"]["max_lr"],
        steps_per_epoch=len(train_dl),
        epochs=config["trainer"]["max_epoch"],
    )

    # training loop
    epochs = config["trainer"]["max_epoch"]

    # best validation loss
    best_val_loss = float("inf")
    stopping_counter = 0
    min_delta = 1e-4

    for epoch in range(0, epochs):
        # train
        pbar = tqdm(train_dl)
        mean_loss = torch.zeros(1, device=device)  # avg loss for epoch
        running_loss = torch.zeros(1, device=device)
        for ib, (xb, yb) in enumerate(pbar):
            m.train()
            xb, yb = xb.to(device), yb.to(device)

            # feed forward
            logits = m(xb)
            b, s, c = logits.shape
            logits = logits.view(b * s, -1)
            loss = F.cross_entropy(logits, yb.view(-1))

            # backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # log
            mean_loss += loss * xb.shape[0]
            running_loss = (ib * running_loss + loss) / (ib + 1)
            pbar.set_description(f"Epoch {epoch}/{epochs}")
            pbar.set_postfix({"train/loss": running_loss.item()})

            if args.wandb and ib % config["wandb"]["log_freq"] == 0:
                wandb.log({"step": epoch * len(train_dl.dataset) + ib})
                wandb.log({"train/loss": loss.item()})
                wandb.log({"lr": scheduler.get_last_lr()[0]})

        # validation
        pbar = tqdm(val_dl)
        mean_loss = torch.zeros(1, device=device)  # avg loss for epoch
        running_loss = torch.zeros(1, device=device)  # running mean batch
        for ib, (xb, yb) in enumerate(pbar):
            m.eval()
            xb, yb = xb.to(device), yb.to(device)
            # feed forward
            with torch.no_grad():
                logits = m(xb)
                b, s, c = logits.shape
                logits = logits.view(b * s, -1)
                loss = F.cross_entropy(logits, yb.view(-1))
                mean_loss += loss * xb.shape[0]

            # log
            running_loss = (ib * running_loss + loss) / (ib + 1)
            pbar.set_postfix({"validation/loss": running_loss.item()})
        mean_loss /= len(val_dl.dataset)
        pbar.set_postfix({"validation/loss": mean_loss.item()})
        if args.wandb:
            wandb.log({"epoch": epoch, "validation/loss": mean_loss.item()})

        # poor man's earlystopping
        if mean_loss - min_delta < best_val_loss:
            best_val_loss = mean_loss
            stopping_counter = 0
            # save best only
            save_checkpoint(
                config["checkpoint"]["path"],
                model=m,
                optimizer=optimizer,
                epoch=epoch,
                loss=mean_loss.item(),
                config=config if isinstance(config, dict) else dict(config),
            )
        elif stopping_counter >= 5:  # epoch
            return
        else:
            stopping_counter += 1

        # save_checkpoint(
        #     config["checkpoint"]["path"],
        #     model=m,
        #     optimizer=optimizer,
        #     epoch=epoch,
        #     loss=mean_loss.item(),
        #     config=config if isinstance(config, dict) else dict(config),
        # )


def save_checkpoint(path, model, optimizer, epoch, loss, config):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    path_save = path / f"epoch={epoch}&validation.loss={loss:.4f}.ckpt"
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": loss,
            "config": config,
        },
        path_save,
    )
    print(f"saving model checkpoint to {path_save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/from_scratch.yaml", required=True
    )
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    main(args)
