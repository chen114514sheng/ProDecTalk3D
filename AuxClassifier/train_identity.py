# AuxClassifier/train_identity.py

import os
import sys
import json
import yaml
import time
import random
import argparse
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 将项目根目录加入路径，保持与仓库现有风格一致
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DataProcess.Dataload import CustomDataset
from AuxClassifier.sequence_classifier import SequenceClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(data_dir: str):
    return CustomDataset(data_dir)


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = True,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


def parse_args():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_config = os.path.join(project_root, "config.yaml")
    default_ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints_identity")

    parser = argparse.ArgumentParser(description="Train auxiliary identity classifier for swap experiments.")
    parser.add_argument("--config", type=str, default=default_config, help="Path to config.yaml")
    parser.add_argument("--train_dir", type=str, default=None, help="Override train dataset path")
    parser.add_argument("--val_dir", type=str, default=None, help="Override val dataset path")
    parser.add_argument("--checkpoint_dir", type=str, default=default_ckpt_dir, help="Directory to save checkpoints")

    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--hidden_dim", type=int, default=256, help="Classifier hidden dim")
    parser.add_argument("--num_heads", type=int, default=4, help="Transformer heads")
    parser.add_argument("--num_layers", type=int, default=3, help="Transformer layers")
    parser.add_argument("--ff_dim", type=int, default=512, help="FFN dim")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--max_len", type=int, default=256, help="Max sequence length for positional encoding")
    parser.add_argument("--use_cls_token", action="store_true", help="Use cls token instead of attention pooling")

    return parser.parse_args()


def get_identity_label(person_one_hot: torch.Tensor) -> torch.Tensor:
    """
    person_one_hot: [B, N_id]
    返回身份类别索引 [B]
    """
    if person_one_hot.dim() != 2:
        raise ValueError(f"person_one_hot should be [B, N_id], but got shape {list(person_one_hot.shape)}")
    return torch.argmax(person_one_hot, dim=1).long()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for _, person_one_hot, _, _, _, _, exp, jaw, mask in loader:
        exp = exp[:, :, :100].float().to(device)
        jaw = jaw.float().to(device)
        mask = mask.float().to(device)
        targets = get_identity_label(person_one_hot).to(device)

        logits = model.forward_from_exp_jaw(
            exp,
            jaw,
            mask=mask,
            return_features=False,
            return_attn=False
        )

        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * targets.size(0)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for _, person_one_hot, _, _, _, _, exp, jaw, mask in loader:
        exp = exp[:, :, :100].float().to(device)
        jaw = jaw.float().to(device)
        mask = mask.float().to(device)
        targets = get_identity_label(person_one_hot).to(device)

        logits = model.forward_from_exp_jaw(
            exp,
            jaw,
            mask=mask,
            return_features=False,
            return_attn=False
        )

        loss = criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * targets.size(0)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def save_checkpoint(
    save_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    args,
    history: Dict,
    num_classes: int,
    label_names: List[str],
):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "num_classes": num_classes,
        "label_type": "identity",
        "label_names": label_names,
        "args": vars(args),
        "history": history,
    }
    torch.save(state, save_path)


def main():
    args = parse_args()
    set_seed(args.seed)

    config = load_yaml(args.config)

    train_dir = args.train_dir if args.train_dir is not None else config["train_file_path"]
    val_dir = args.val_dir if args.val_dir is not None else config["val_file_path"]

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    train_dataset = build_dataset(train_dir)
    val_dataset = build_dataset(val_dir)

    if not hasattr(train_dataset, "person_ids"):
        raise AttributeError("CustomDataset does not have attribute 'person_ids'.")

    label_names = list(train_dataset.person_ids)
    num_classes = len(label_names)

    model = SequenceClassifier(
        input_dim=103,                 # exp(100) + jaw(3)
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        max_len=args.max_len,
        use_cls_token=args.use_cls_token,
    ).to(device)

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_epoch = -1
    start_time = time.time()

    print("=" * 80)
    print("Start training auxiliary identity classifier")
    print(f"train_dir      : {train_dir}")
    print(f"val_dir        : {val_dir}")
    print(f"checkpoint_dir : {args.checkpoint_dir}")
    print(f"device         : {device}")
    print(f"num_classes    : {num_classes}")
    print(f"label_names    : {label_names}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.6f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.4f}"
        )

        last_ckpt_path = os.path.join(args.checkpoint_dir, "model_last.pth")
        save_checkpoint(
            save_path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_acc=best_val_acc,
            args=args,
            history=history,
            num_classes=num_classes,
            label_names=label_names,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_ckpt_path = os.path.join(args.checkpoint_dir, "model_best.pth")
            save_checkpoint(
                save_path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                args=args,
                history=history,
                num_classes=num_classes,
                label_names=label_names,
            )
            print(f"  -> Save best checkpoint to {best_ckpt_path}")

    total_time = time.time() - start_time

    summary = {
        "label_type": "identity",
        "label_names": label_names,
        "num_classes": num_classes,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "total_time_sec": total_time,
        "train_dir": train_dir,
        "val_dir": val_dir,
        "args": vars(args),
    }

    with open(os.path.join(args.checkpoint_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.checkpoint_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Training finished.")
    print(f"Best epoch   : {best_epoch}")
    print(f"Best val acc : {best_val_acc:.4f}")
    print(f"Time (sec)   : {total_time:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()