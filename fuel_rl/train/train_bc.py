"""行为克隆训练."""
import os
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fuel_rl.models import ViewpointHead, Encoder3D
from fuel_rl.data.dataset import make_dataloaders
from fuel_rl.config import *


def train_bc(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # 数据
    train_loader, val_loader = make_dataloaders(
        args.data_path, batch_size=args.batch_size, val_split=BC_VAL_SPLIT,
    )

    # 模型
    encoder = Encoder3D(grid_size=args.grid_size, channels=ENCODER_CHANNELS, embed_dim=ENCODER_EMBED_DIM,
                        input_shape=(args.grid_size, args.grid_size, args.grid_z))
    model = ViewpointHead(encoder, embed_dim=ENCODER_EMBED_DIM).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=BC_WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    start_epoch = 0
    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    # 断点续训
    if args.resume:
        ckpt = torch.load(args.resume, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "history" in ckpt:
            history = ckpt["history"]
        print(f"Resumed from epoch {start_epoch}")

    # 训练
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        n_batch = 0
        for grids, targets in train_loader:
            grids, targets = grids.to(DEVICE), targets.to(DEVICE)
            pred = model(grids)
            loss = criterion(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batch += 1

        train_loss /= max(n_batch, 1)

        # 验证
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for grids, targets in val_loader:
                grids, targets = grids.to(DEVICE), targets.to(DEVICE)
                pred = model(grids)
                val_loss += criterion(pred, targets).item()
                n_val += 1
        val_loss /= max(n_val, 1)

        scheduler.step(val_loss)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        # 保存
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "history": history,
        }, os.path.join(args.save_dir, "latest_checkpoint.pth"))

        if epoch % 5 == 0 or is_best:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}: train={train_loss:.6f} val={val_loss:.6f} "
                  f"lr={lr:.1e} {'*' if is_best else ''}")

    # 绘图
    _plot_history(history, args.save_dir)
    print(f"BC training done. Best val loss: {best_val_loss:.6f}")


def _plot_history(history, save_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("BC Training")
    ax.legend()
    ax.set_yscale("log")
    fig.savefig(os.path.join(save_dir, "bc_loss.png"), dpi=150)
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="BC Training")
    parser.add_argument("--data-path", type=str, default=COLLECT_SAVE_PATH)
    parser.add_argument("--save-dir", type=str, default=BC_SAVE_DIR)
    parser.add_argument("--epochs", type=int, default=BC_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BC_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=BC_LR)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--grid-z", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train_bc(args)


if __name__ == "__main__":
    main()
