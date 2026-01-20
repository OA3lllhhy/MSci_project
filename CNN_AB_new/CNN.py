#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, accuracy_score, average_precision_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import trange

# Vesion 2.0 data directory
# /ceph/submit/data/user/h/haoyun22/Patches_CNN_Data/AB_patches_final_2.npz
# Note: V2.0 has 2 channels (energy deposited in Layer A and B)
# Note: V2.1 has 4 channels (energy deposited, multiplicity in Layer A and B)

# Version 3 data directory
# /ceph/submit/data/user/h/haoyun22/Patches_CNN_Data/AB_patches_V3.npz
# Note: V3 has 6 channels (energy deposited, multiplicity, extent in Layer A and B)

parser = argparse.ArgumentParser()
parser.add_argument("--classify", action="store_true")
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--data", default="/ceph/submit/data/user/h/haoyun22/Patches_CNN_Data/AB_patches_final_2.npz")
parser.add_argument("--exp_name", default="1", help="Experiment name identifier")
parser.add_argument("--repeat", help="Train model multiple times with different seeds for averaging", action="store_true")
args = parser.parse_args()


class EarlyStopping:
    def __init__(self, patience=20, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# # exp_name = args.exp_name
# outdir = "/work/submit/haoyun22/FCC-Beam-Background/CNN_AB_new"
outdir_split_data = "/ceph/submit/data/user/h/haoyun22/Patches_CNN_Data/CNN_split_data"

# ======================================================================
# CLASSIFY
# ======================================================================
if args.classify:

    # Disable problematic modules
    sys.modules['onnx'] = None
    sys.modules['onnxruntime'] = None
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["TORCH_DISABLE_CPP_PROTOS"] = "1"
    os.environ["TORCH_USE_RTLD_GLOBAL"] = "0"


    exp_name = args.exp_name

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    outdir = f"/work/submit/haoyun22/FCC-Beam-Background/CNN_AB_new/{exp_name}"
    os.makedirs(outdir, exist_ok=True)
    outdir_split_data = "/ceph/submit/data/user/h/haoyun22/Patches_CNN_Data/CNN_split_data"
    os.makedirs(outdir_split_data, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    print("Loading dataset...")
    npz = np.load(args.data)
    X = npz["X"].astype(np.float32)      # (N, 2, H, W) 2 stands for in H x W pixels image, the energy deposits in Layer A and Layer B
    y = npz["y"].astype(np.float32)      # (N,)

    # --- simple normalisation ---
    # X /= np.max(X)

    mean = X.mean(axis=(0,2,3), keepdims=True)
    std = X.std(axis=(0,2,3), keepdims=True) + 1e-6
    X = (X - mean) / std

    # Convert to torch
    X_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    N_pos = y_torch.sum()
    N_neg = y_torch.shape[0] - N_pos
    class_weight = N_neg / N_pos
    class_weight = torch.tensor([class_weight], dtype=torch.float32).to(device)

    dataset = TensorDataset(X_torch, y_torch)

    dataset_train, dataset_validate, dataset_test = random_split(
        dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(seed)
    )

    dloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dloader_validate = DataLoader(dataset_validate, batch_size=128, shuffle=False)

    patch_size = X.shape[2]
    print(f"Detected patch size: {patch_size} x {patch_size}")
    feature_map_size = patch_size // 4  # After 2x MaxPool2d with kernel_size=2
    flatten_size = 32 * feature_map_size * feature_map_size

    # ======================================================================
    # CNN MODEL (same style as notes: Sequential + simple)
    # ======================================================================
    model = nn.Sequential(
    nn.Conv2d(2, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(0.1),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(0.1),

    nn.Flatten(),
    nn.Linear(flatten_size, 1),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    loss_fcn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weight).to(device))
    #loss_fcn = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    # ======================================================================
    # TRAINING LOOP (same simple style)
    # ======================================================================
    tloss, vloss = [], []

    def train_epoch():
        model.train()
        total = 0

        for Xb, yb in dloader_train:
            Xb = Xb.to(device)
            yb = yb.to(device)

            pred = model(Xb)
            loss = loss_fcn(pred, yb)

            opt.zero_grad()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            opt.step()

            total += loss.item()

        # Validation
        model.eval()
        vtotal = 0
        with torch.no_grad():
            for Xv, yv in dloader_validate:
                Xv = Xv.to(device)
                yv = yv.to(device)
                pred_v = model(Xv)
                vloss = loss_fcn(pred_v, yv)
                vtotal += vloss.item()

        return total/len(dloader_train), vtotal/len(dloader_validate)

    print(f"Training CNN for experiment {exp_name}...")
    for epoch in trange(200, desc="Epochs"):
        train_l, val_l = train_epoch()
        tloss.append(train_l)
        vloss.append(val_l)

    np.save(f"{outdir}/CNN_losses_{exp_name}.npy", np.array([tloss, vloss]))
    torch.save(model, f"{outdir}/CNN_model_{exp_name}.pt")

    # Save train/val/test for evaluation
    torch.save({
        "train": dataset_train,
        "validate": dataset_validate,
        "test": dataset_test
    }, f"{outdir_split_data}/CNN_split_{exp_name}.pt")

    print("Training complete.")


# ======================================================================
# EVALUATE
# ======================================================================
if args.evaluate:

    # Disable problematic modules
    sys.modules['onnx'] = None
    sys.modules['onnxruntime'] = None
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["TORCH_DISABLE_CPP_PROTOS"] = "1"
    os.environ["TORCH_USE_RTLD_GLOBAL"] = "0"

    exp_name = args.exp_name
    outdir = f"/work/submit/haoyun22/FCC-Beam-Background/CNN_AB_new/{exp_name}"
    os.makedirs(outdir, exist_ok=True)

    # Load model + losses + splits
    model = torch.load(f"{outdir}/CNN_model_{exp_name}.pt", weights_only=False, map_location="cpu")
    model = model.cpu()
    model.eval()

    tloss, vloss = np.load(f"{outdir}/CNN_losses_{exp_name}.npy")
    data = torch.load(f"{outdir_split_data}/CNN_split_{exp_name}.pt", weights_only=False, map_location="cpu")
    test_set = data["test"]

    X_test = torch.stack([x for x, _ in test_set])
    y_test = torch.stack([y for _, y in test_set])

    # Loss curves
    fig, ax = plt.subplots(figsize=(8,6), dpi=150)
    ax.plot(tloss, label="Training loss", color="black")
    ax.plot(vloss, label="Validation loss", color="#D55E00")
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Binary cross entropy", fontsize=16)
    ax.set_title("CNN training loss", fontsize=20)
    ax.tick_params(labelsize=12, which="both", top=True, right=True, direction="in")
    ax.grid(color="xkcd:dark blue", alpha=0.2)
    ax.legend()
    plt.savefig(f"{outdir}/CNN_losses_{exp_name}.png")
    plt.close()

    # Predictions
    with torch.no_grad():
        preds = model(X_test).cpu()
        labels = y_test.cpu()
        preds_bin = (preds >= 0.5).float()

    acc = accuracy_score(labels, preds_bin)
    print(f"Test accuracy of experiment {exp_name} = {acc*100:.2f}%")

    # ROC curve
    fpr, tpr, _ = roc_curve(labels.numpy(), preds.numpy())
    auc = roc_auc_score(labels.numpy(), preds.numpy())
    pr_auc = average_precision_score(labels.numpy(), preds.numpy())
    print(f"ROC AUC of experiment {exp_name} = {auc}")
    print(f"PR AUC of experiment {exp_name} = {pr_auc}")

    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    ax.plot(fpr, tpr, color="#D55E00", label=f"AUC={auc:.4f}")
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel("False positive rate", fontsize=15)
    ax.set_ylabel("True positive rate", fontsize=15)
    ax.set_title("CNN ROC Curve", fontsize=18)
    ax.grid(alpha=0.3)
    plt.legend()
    plt.savefig(f"{outdir}/CNN_ROC_{exp_name}.png")
    plt.close()

    # Plot PR curve
    precision, recall, _ = precision_recall_curve(labels.numpy(), preds.numpy())
    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    ax.plot(recall, precision, color="#0072B2", label=f"PR AUC={pr_auc:.4f}")
    ax.set_xlabel("Recall", fontsize=15)
    ax.set_ylabel("Precision", fontsize=15)
    ax.set_title("CNN Precision-Recall Curve", fontsize=18)
    ax.grid(alpha=0.3)
    plt.legend()
    plt.savefig(f"{outdir}/CNN_PR_{exp_name}.png")
    plt.close()

if args.repeat:
    import pandas as pd
    from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score
    import random

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    n_repeats = 200
    results = []
    
    # 确保输出目录存在
    exp_name = "20"
    os.makedirs(outdir, exist_ok=True)
    csv_path = f"{outdir}/repeat_results_{exp_name}.csv"
    checkpoint_path = f"/ceph/submit/data/user/h/haoyun22/CNN_test_checkpoint/CNN_train_checkpoints_{exp_name}"
    os.makedirs(checkpoint_path, exist_ok=True)

    # 加载原始数据 (只需加载一次)
    print("Loading dataset for repeated trials...")
    npz = np.load(args.data)
    X_raw = npz["X"].astype(np.float32)
    y_raw = npz["y"].astype(np.float32)
    
    # 标准化 (Channel-wise)
    mean = X_raw.mean(axis=(0,2,3), keepdims=True)
    std = X_raw.std(axis=(0,2,3), keepdims=True) + 1e-6
    X_norm = (X_raw - mean) / std

    for i in range(n_repeats):
        # 1. 设置随机种子
        seed = random.randint(0, 10000000)
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n>>> Trial {i+1}/{n_repeats} | Seed: {seed}")

        # 2. 准备数据
        X_torch = torch.tensor(X_norm, dtype=torch.float32)
        y_torch = torch.tensor(y_raw, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(X_torch, y_torch)
        
        # 划分数据 (每次随机划分)
        train_set, val_set, test_set = random_split(dataset, [0.6, 0.2, 0.2])
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

        # 3. 初始化模型、损失函数和优化器
        # (使用你之前的模型结构)
        model = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 1),
        ).to(device)

        # 计算权重以处理类别不平衡
        n_pos = y_raw.sum()
        class_weight = torch.tensor([(len(y_raw) - n_pos) / n_pos]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        
        early_stopping = EarlyStopping(patience=20, path=f"{checkpoint_path}/temp_checkpoint.pt")

        # 4. 训练循环
        for epoch in range(200):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()

            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for Xv, yv in val_loader:
                    Xv, yv = Xv.to(device), yv.to(device)
                    val_loss += criterion(model(Xv), yv).item()
            
            avg_val_loss = val_loss / len(val_loader)
            early_stopping(avg_val_loss, model)
            
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        # 5. 测试评估 (加载最佳权重)
        model.load_state_dict(torch.load(f"{checkpoint_path}/temp_checkpoint.pt"))
        model.eval()
        
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for Xt, yt in test_loader:
                Xt = Xt.to(device)
                outputs = torch.sigmoid(model(Xt)) # 转为概率
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(yt.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        binary_preds = (all_preds >= 0.5).astype(float)

        # 6. 计算指标
        metrics = {
            "trial": i + 1,
            "seed": seed,
            "accuracy": accuracy_score(all_labels, binary_preds),
            "precision": precision_score(all_labels, binary_preds, zero_division=0),
            "recall": recall_score(all_labels, binary_preds),
            "auc": roc_auc_score(all_labels, all_preds),
            "pr_auc": average_precision_score(all_labels, all_preds)
        }
        results.append(metrics)
        print(f"Trial Result: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")

        # 实时保存到 CSV 以防程序中断
        pd.DataFrame(results).to_csv(csv_path, index=False)

        if (i+1) % 5 == 0:
            print(f"--- Completed {i+1} trials. Current Avg PR-AUC: {np.mean([r['pr_auc'] for r in results]):.4f} ---")

    print(f"\nAll {n_repeats} trials complete. Results saved to {csv_path}")
