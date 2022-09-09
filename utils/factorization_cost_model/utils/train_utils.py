import copy
import math
import os
import random
import time


import torch
import wandb
from torch.optim.lr_scheduler import OneCycleLR

from utils.config import *

from tqdm import tqdm


def mape_criterion(inputs, targets):
    eps = 1e-5
    return 100 * torch.mean(torch.abs(targets - inputs) / (targets + eps))


def train_model(
    config,
    model,
    criterion,
    optimizer,
    max_lr,
    dataloader,
    num_epochs=100,
    log_every=5,
    logger=None,
    train_device="cpu",
):
    since = time.time()
    losses = []
    train_loss = 0
    best_loss = math.inf
    best_model = None
    hash = random.getrandbits(16)
    dataloader_size = {"train": 0, "val": 0}
    for item in dataloader["train"]:
        label = item[1]
        dataloader_size["train"] += label.shape[0]
    for item in dataloader["val"]:
        label = item[1]
        dataloader_size["val"] += label.shape[0]

    model = model.to(train_device)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(dataloader["train"]),
        epochs=num_epochs,
    )
    for epoch in range(num_epochs):
        epoch_start = time.time()

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            pbar = tqdm(dataloader[phase])
            for inputs, labels in pbar:
                original_device = labels.device
                inputs = (
                    inputs[0],
                    inputs[1].to(train_device),
                    inputs[2].to(train_device),
                )
                labels = labels.to(train_device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    # print(outputs.get_device(),labels.get_device())
                    assert outputs.shape == labels.shape
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                pbar.set_description("Loss: {:.3f}".format(loss.item()))
                running_loss += loss.item() * labels.shape[0]
                labels = labels.to(original_device)
                epoch_end = time.time()

            epoch_loss = running_loss / dataloader_size[phase]

            if phase == "val":
                losses.append((train_loss, epoch_loss))
                if epoch_loss <= best_loss:
                    best_loss = epoch_loss
                    best_model = copy.deepcopy(model)
                    saved_model_path = os.path.join(config.experiment.base_path, "weights/")
                    if not os.path.exists(saved_model_path):
                        os.makedirs(saved_model_path)
                    full_path = os.path.join(
                            saved_model_path,
                            f"best_model_{config.experiment.name}_{hash:4x}.pt",
                        )
                    logger.debug(f"Saving checkpoint to {full_path}")
                    torch.save(
                        model.state_dict(),
                        full_path,
                    )

                if config.wandb.use_wandb:
                    wandb.log(
                        {
                            "best_msle": best_loss,
                            "train_msle": train_loss,
                            "val_msle": epoch_loss,
                            "epoch": epoch,
                        }
                    )
                print(
                    "Epoch {}/{}:  "
                    "train Loss: {:.4f}   "
                    "val Loss: {:.4f}   "
                    "time: {:.2f}s   "
                    "best: {:.4f}".format(
                        epoch + 1,
                        num_epochs,
                        train_loss,
                        epoch_loss,
                        epoch_end - epoch_start,
                        best_loss,
                    )
                )
                if epoch % log_every == 0:
                    logger.info(
                        "Epoch {}/{}:  "
                        "train Loss: {:.4f}   "
                        "val Loss: {:.4f}   "
                        "time: {:.2f}s   "
                        "best: {:.4f}".format(
                            epoch + 1,
                            num_epochs,
                            train_loss,
                            epoch_loss,
                            epoch_end - epoch_start,
                            best_loss,
                        )
                    )
            else:
                train_loss = epoch_loss
                scheduler.step()
    time_elapsed = time.time() - since

    print(
        "Training complete in {:.0f}m {:.0f}s   "
        "best validation loss: {:.4f}".format(
            time_elapsed // 60, time_elapsed % 60, best_loss
        )
    )
    with open(log_file, "a+") as f:
        f.write(
            "-----> Training complete in {:.0f}m {:.0f}s   "
            "best validation loss: {:.4f}\n ".format(
                time_elapsed // 60, time_elapsed % 60, best_loss
            )
        )

    return losses, best_model
