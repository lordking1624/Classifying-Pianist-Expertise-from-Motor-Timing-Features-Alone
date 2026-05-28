import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import Model

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED = 22012681
BATCH_SIZE = 16
EPOCHS = 300
LR = 3e-4
PATIENCE = 20
HIDDEN_SIZE = 128
BIDIRECTIONAL = True
NUM_LAYERS = 2
DROPOUT = 0.0
INPUT_SIZE = 4
MANIFEST_PATH = "Data/processed/manifest.csv"
NUM_WORKERS = 0
REMOVE = ["ioi"]
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 2
SCHEDULER_CYCLE = 20
SCHEDULER_MIN = 1e-6

print(
    f"CONFIG | SEED={SEED} | BATCH_SIZE={BATCH_SIZE} | EPOCHS={EPOCHS} | "
    f"LR={LR} | PATIENCE={PATIENCE} | HIDDEN_SIZE={HIDDEN_SIZE} | "
    f"BIDIRECTIONAL={BIDIRECTIONAL} | NUM_LAYERS={NUM_LAYERS} | "
    f"DROPOUT={DROPOUT} | INPUT_SIZE={INPUT_SIZE} | "
    f"MANIFEST_PATH='{MANIFEST_PATH}' | NUM_WORKERS={NUM_WORKERS} | "
    f"REMOVE='{REMOVE}' | SCHEDULER_PATIENCE='{SCHEDULER_PATIENCE}' | "
    f"SCHEDULER_FACTOR='{SCHEDULER_FACTOR}' | SCHEDULER_MIN='{SCHEDULER_MIN}' | "
    f"SCHEDULER_CYCLE='{SCHEDULER_CYCLE}' | "
)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model = Model(
    input_size=INPUT_SIZE,
    bidirectional=BIDIRECTIONAL,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=SCHEDULER_CYCLE,
    T_mult=SCHEDULER_FACTOR,
    eta_min=SCHEDULER_MIN,
)
print("Grabbing Data...")
train_loader, val_loader, test_loader = get_dataloaders(
    manifest_path=MANIFEST_PATH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    remove=REMOVE,
)
print("Done")


def train(
    model,
    optimizer,
    criterion,
    scheduler,
    train_loader,
    val_loader,
    epochs,
    device,
    patience,
):
    print("Training ...")
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    best_val_loss = 1.0
    best_val_acc = 0.0
    best_state_dict = None

    epochs_without_improve = 0
    for epoch in range(epochs):
        train_total_loss = 0.0
        train_correct = 0
        train_total = 0

        model.train()
        for _, (arrays, labels) in enumerate(train_loader):
            arrays = arrays.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(arrays)

            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            train_total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_total_loss / len(train_loader)
        train_acc = train_correct / train_total

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_total_loss = 0.0
        val_correct = 0
        val_total = 0

        model.eval()
        with torch.no_grad():
            for arrays, labels in val_loader:
                arrays = arrays.to(device)
                labels = labels.to(device)

                logits = model(arrays)
                loss = criterion(logits, labels)
                val_total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_total_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Improved "
            )
        else:
            epochs_without_improve += 1

            if epochs_without_improve >= patience:
                print(
                    f"Early Stopping After | Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )
                break
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )
        scheduler.step(epoch)

    model.load_state_dict(best_state_dict)
    return train_losses, train_accs, val_losses, val_accs, best_val_loss, best_val_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for arrays, labels in loader:
            arrays = arrays.to(device)
            labels = labels.to(device)
            logits = model(arrays)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc = total_correct / total_samples

    # F1 (binary, positive class = 1)
    tp = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(all_preds, all_labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(all_preds, all_labels))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return avg_loss, acc, f1, all_preds, all_labels


import json
import os
from datetime import datetime

import torch


def save_run(
    model,
    config,
    best_val_loss,
    best_val_acc,
    test_loss,
    test_acc,
    test_f1,
    train_losses,
    train_accs,
    val_losses,
    val_accs,
    test_preds,
    test_labels,
    base_dir="Runs",
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remove_tag = "full" if not config["REMOVE"] else "-".join(config["REMOVE"])
    dir_name = (
        f"{timestamp}"
        f"_LR{config['LR']}"
        f"_BS{config['BATCH_SIZE']}"
        f"_H{config['HIDDEN_SIZE']}"
        f"_L{config['NUM_LAYERS']}"
        f"_DO{config['DROPOUT']}"
        f"_remove-{remove_tag}"
        f"_valLoss{best_val_loss:.4f}"
        f"_valAcc{best_val_acc:.4f}"
        f"_testAcc{test_acc:.4f}"
    )

    run_dir = os.path.join(base_dir, dir_name)
    os.makedirs(run_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    results = {
        **config,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "test_preds": test_preds,
        "test_labels": test_labels,
        "timestamp": timestamp,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Run saved -> {run_dir}")
    return run_dir


if __name__ == "__main__":
    train_losses, train_accs, val_losses, val_accs, best_val_loss, best_val_acc = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        device=device,
        patience=PATIENCE,
    )

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    print(
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}"
    )

    config = {
        "SEED": SEED,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "PATIENCE": PATIENCE,
        "HIDDEN_SIZE": HIDDEN_SIZE,
        "BIDIRECTIONAL": BIDIRECTIONAL,
        "NUM_LAYERS": NUM_LAYERS,
        "DROPOUT": DROPOUT,
        "INPUT_SIZE": INPUT_SIZE,
        "MANIFEST_PATH": MANIFEST_PATH,
        "REMOVE": REMOVE,
        "SCHEDULER_CYCLE": SCHEDULER_CYCLE,
    }

    run_dir = save_run(
        model,
        config,
        best_val_loss,
        best_val_acc,
        test_loss,
        test_acc,
        test_f1,
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        test_preds,
        test_labels,
    )
