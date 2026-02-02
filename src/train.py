# - SageMaker PyTorch training script (classification only: cat vs non-cat)
# - Reads manifest CSVs from channels, loads images from S3 on-the-fly
# - Resizes all images; augments only training cats (label==1)
# - Trains a small CNN and prints real per-epoch metrics to logs
# - Saves model weights to SM_MODEL_DIR so SageMaker uploads artifacts

import os
import io
import boto3
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------- S3 image loader ----------
_s3 = boto3.client("s3")

def read_image_from_image_id(image_id: str) -> Image.Image:
    bucket, key =image_id.replace("s3://", "").split("/", 1)
    obj = _s3.get_object(Bucket=bucket, Key=key)
    return Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")

# ---------- Dataset ----------
class CatClsDataset(Dataset):
    # Comments:
    # - Resizes ALL images to image_size
    # - Augments ONLY if is_train=True and label==1
    def __init__(self, df: pd.DataFrame, image_size: int = 224, is_train: bool = False):
        self.df = df.reset_index(drop=True)
        self.image_size = int(image_size)
        self.is_train = bool(is_train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        

        # TODO(only if your column names differ):
        image_id = row["image_id"]
        label = int(row["label"])

        img = read_image_from_image_id(image_id)
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Augment ONLY cat images during training
        if self.is_train and label == 1:
            if np.random.rand() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() < 0.3:
                arr = np.array(img).astype(np.float32)
                arr = np.clip(arr * (0.8 + 0.4 * np.random.rand()), 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)

        # To tensor (CHW) and normalize [0,1]
        arr = np.array(img).astype(np.float32) / 255.0  # HWC
        x = torch.from_numpy(arr).permute(2, 0, 1)      # CHW
        y = torch.tensor(label, dtype=torch.long)

        return x, y

# ---------- Model ----------
class SmallCNN(nn.Module):
    #  Small CNN that runs on CPU (ml.m5.large) and gives real metrics
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

# ---------- Train/Val loop ----------
def run_one_epoch(model, loader, criterion, optimizer, device, train_mode: bool):
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(train_mode):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == yb).sum().item())
            total += xb.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc

def main():
    # SageMaker channels
    train_dir = os.environ.get("SM_CHANNEL_TRAIN")
    val_dir = os.environ.get("SM_CHANNEL_VAL")
    model_dir = os.environ.get("SM_MODEL_DIR", ".")

    # Hyperparameters passed from estimator
    epochs = int(os.environ.get("SM_HP_EPOCHS", "2"))
    batch_size = int(os.environ.get("SM_HP_BATCH_SIZE", "32"))
    lr = float(os.environ.get("SM_HP_LR", "0.001"))
    image_size = int(os.environ.get("SM_HP_IMAGE_SIZE", "224"))

    print("Train dir:", train_dir)
    print("Val dir:", val_dir)
    print("Model dir:", model_dir)
    print("HP:", {"epochs": epochs, "batch_size": batch_size, "lr": lr, "image_size": image_size})

    # Load manifests
    train_csv = os.path.join(train_dir, "train.csv")
    val_csv = os.path.join(val_dir, "val.csv")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    print("Train rows:", len(train_df))
    print("Val rows:", len(val_df))

    # Datasets / loaders
    train_ds = CatClsDataset(train_df, image_size=image_size, is_train=True)
    val_ds = CatClsDataset(val_df, image_size=image_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Device:", device)

    # Train
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
        va_loss, va_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train_mode=False)

        print(
            f"epoch={epoch} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )

    # Save model weights (SageMaker uploads anything in SM_MODEL_DIR)
    out_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), out_path)
    print("Saved model:", out_path)

if __name__ == "__main__":
    main()
