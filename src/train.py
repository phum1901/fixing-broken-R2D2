import argparse

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.callbacks import EarlyStopping, ModelCheckpoint
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
    train_dl = DataLoader(dataset=train_ds, batch_size=config["trainer"]["batch_size"], shuffle=True)
    val_dl = DataLoader(dataset=val_ds, batch_size=config["trainer"]["batch_size"], shuffle=False)

    # init model
    config["model"]["vocab_size"] = len(dataset.chars)
    config["model"]["chars"] = dataset.chars
    m = Model(**config["model"])
    m.to(device)
    if args.wandb:
        wandb.watch(models=m, log_freq=config["wandb"]["log_freq"])  # log every n batch

    # optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=float(config["optimizer"]["lr"]))

    # scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(config["scheduler"]["max_lr"]),
        steps_per_epoch=len(train_dl),
        epochs=config["trainer"]["max_epoch"],
    )

    # callbacks
    early_stopping = EarlyStopping(
        patience=int(config["callback"]["patience"]), min_delta=float(config["callback"]["min_delta"])
    )
    model_checkpoint = ModelCheckpoint(config["checkpoint"]["path"])

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

        # callbacks
        save_path = model_checkpoint(
            mean_loss.item(),
            model=m,
            optimizer=optimizer,
            epoch=epoch,
            config=config if isinstance(config, dict) else dict(config),
        )
        if args.wandb and save_path is not None:
            art = wandb.Artifact(f"{wandb.run.id}", type="model")
            art.add_file(save_path)
            wandb.log_artifact(art, aliases=["latest"])

        if early_stopping(mean_loss.item()):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/from_scratch.yaml", required=True)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    main(args)
