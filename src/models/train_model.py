import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from collections import Counter
import numpy as np

def main(data_dir="data/trashnet_split", model_dir="models", epochs=30, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------------
    # 🔹 Data Augmentation
    # --------------------------
    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(Path(data_dir)/"train", transform=transform_train)
    val_ds   = datasets.ImageFolder(Path(data_dir)/"val", transform=transform_val)
    test_ds  = datasets.ImageFolder(Path(data_dir)/"test", transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # --------------------------
    # 🔹 Model: ResNet18 (fine-tuned)
    # --------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # unfreeze ALL layers for fine-tuning
    for param in model.parameters():
        param.requires_grad = True
    # replace final layer
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(device)

    # --------------------------
    # 🔹 Class Weights (to fix imbalance)
    # --------------------------
    counts = Counter(train_ds.targets)  # dict: {class_idx: count}
    weights = [1.0 / counts[i] for i in range(len(train_ds.classes))]
    weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # --------------------------
    # 🔹 Optimizer + Scheduler
    # --------------------------
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # --------------------------
    # 🔹 Training Loop
    # --------------------------
    best_acc = 0.0
    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # ---- Validation ----
        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss /= total
        val_acc = correct / total

        print(f"Epoch {epoch+1}/{epochs} | Train Acc={train_acc:.3f} | Val Acc={val_acc:.3f}")

        scheduler.step(val_acc)

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "classes": train_ds.classes
            }, Path(model_dir)/"resnet_trashnet.pth")
            print("✅ Saved best model")

    # --------------------------
    # 🔹 Final Test Report
    # --------------------------
    ckpt = torch.load(Path(model_dir)/"resnet_trashnet.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            y_true += labels.cpu().tolist()
            y_pred += preds.cpu().tolist()
    print("\n📊 Test Report:\n", classification_report(y_true, y_pred, target_names=ckpt["classes"]))

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",default="data/trashnet_split")
    ap.add_argument("--model-dir",default="models")
    ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--batch-size",type=int,default=32)
    ap.add_argument("--lr",type=float,default=1e-4)
    args=ap.parse_args()
    main(args.data_dir,args.model_dir,args.epochs,args.batch_size,args.lr)
