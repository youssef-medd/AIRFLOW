import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, PATIENCE, DEFAULT_CV_MODEL
from cv_model import build_model
from cv_dataloader import get_loaders
from early_stopping import EarlyStopping


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(imgs)
        correct    += (logits.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * len(imgs)
        correct    += (logits.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def train(data_root: str, arch: str = "resnet18", save_path: str = DEFAULT_CV_MODEL,
          patience: int = PATIENCE):
    train_dl, val_dl = get_loaders(data_root, batch_size=BATCH_SIZE)
    model     = build_model(arch).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    stopper   = EarlyStopping(patience=patience, save_path=save_path)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, criterion)
        vl_loss, vl_acc = evaluate(model, val_dl, criterion)
        scheduler.step()
        improved = stopper.best_score is None or vl_acc > stopper.best_score + stopper.min_delta
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val_loss={vl_loss:.4f} acc={vl_acc:.3f}" + (" *" if improved else ""))
        if stopper.step(vl_acc, model):
            print(f"Early stop at epoch {epoch}.")
            break

    print(f"Best val acc: {stopper.best_score:.3f} — model saved to {save_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data",  required=True)
    p.add_argument("--arch",  default="resnet18")
    p.add_argument("--out",      default=DEFAULT_CV_MODEL)
    p.add_argument("--patience", type=int, default=PATIENCE)
    args = p.parse_args()
    train(args.data, args.arch, args.out, args.patience)
