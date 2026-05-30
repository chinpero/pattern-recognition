import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import HipKeypointDataset
from src.model import HeatmapKeypointModel


def heatmap_loss(pred_heatmaps, target_heatmaps, hm_mask):
    """MSE loss on heatmaps, only for keypoints present in the annotation."""
    diff = (pred_heatmaps - target_heatmaps) ** 2
    masked_diff = diff * hm_mask[:, :, None, None]
    return masked_diff.sum() / (hm_mask.sum() * pred_heatmaps.shape[-2] * pred_heatmaps.shape[-1] + 1e-6)


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Train", leave=False):
        images = batch["image"].to(device)
        target_hm = batch["heatmap"].to(device)
        hm_mask = batch["hm_mask"].to(device)

        optimizer.zero_grad()
        pred_hm = model(images)
        loss = heatmap_loss(pred_hm, target_hm, hm_mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validate", leave=False):
            images = batch["image"].to(device)
            target_hm = batch["heatmap"].to(device)
            hm_mask = batch["hm_mask"].to(device)

            pred_hm = model(images)
            loss = heatmap_loss(pred_hm, target_hm, hm_mask)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDH hip keypoint model")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)
    parser.add_argument("--mask-sensitive", action="store_true")
    parser.add_argument("--num-keypoints", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = HipKeypointDataset(
        data_dir=args.train_dir,
        image_size=args.image_size,
        heatmap_sigma=args.heatmap_sigma,
        mask_sensitive=args.mask_sensitive,
        is_train=True,
    )
    val_dataset = HipKeypointDataset(
        data_dir=args.val_dir,
        image_size=args.image_size,
        heatmap_sigma=args.heatmap_sigma,
        mask_sensitive=args.mask_sensitive,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    model = HeatmapKeypointModel(
        num_keypoints=args.num_keypoints, pretrained=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss: {train_loss:.6f}  "
            f"val_loss: {val_loss:.6f}  "
            f"lr: {lr:.2e}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / "best_checkpoint.pt")
            print(f"  -> Best model saved (val_loss: {val_loss:.6f})")

    print(f"Training complete. Best model: {output_dir / 'best_checkpoint.pt'}")


if __name__ == "__main__":
    main()
