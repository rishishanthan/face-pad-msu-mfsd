import argparse
import os
import random

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def laplacian_channel(rgb):
    """
    Simple frequency/high-pass cue using Laplacian of grayscale.
    This often helps print/replay detection.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap = lap - lap.min()
    if lap.max() > 1e-6:
        lap = lap / lap.max()
    lap = (lap * 255.0).astype(np.uint8)
    return lap


class FrameDS(Dataset):
    """
    Loads frame paths from CSV and returns tensor + label.
    label: 1 = live, 0 = attack
    """

    def __init__(self, csv_path, use_freq=False):
        self.df = pd.read_csv(csv_path)
        self.use_freq = use_freq

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        bgr = cv2.imread(r["frame_path"])
        if bgr is None:
            bgr = np.zeros((224, 224, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224))

        if self.use_freq:
            lap = laplacian_channel(rgb)
            x = np.concatenate([rgb, lap[..., None]], axis=2)  # 4 channels
        else:
            x = rgb  # 3 channels

        x = torch.from_numpy(x).float() / 255.0
        x = x.permute(2, 0, 1).contiguous()

        y = torch.tensor(float(r["label"]))  # 1.0 or 0.0
        return x, y


def adapt_first_conv(m, in_ch):
    """
    If using 4 channels, modify the first conv layer to accept 4 input channels.
    - copy existing RGB weights
    - create 4th channel weights as mean of RGB weights
    """
    conv = m.features[0][0]
    if conv.in_channels == in_ch:
        return m

    new = nn.Conv2d(
        in_ch,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=False,
    )

    with torch.no_grad():
        w = conv.weight  # (out_ch, 3, k, k)
        mean = w.mean(dim=1, keepdim=True)  # (out_ch, 1, k, k)
        # concatenate RGB weights + mean channel as 4th
        new.weight.copy_(torch.cat([w, mean.repeat(1, in_ch - 3, 1, 1)], dim=1))

    m.features[0][0] = new
    return m


class MobileNetPAD(nn.Module):
    """
    MobileNetV2 backbone + 1-unit sigmoid head for PAD binary classification.
    weights=None avoids downloading pretrained weights (no SSL issues).
    """

    def __init__(self, in_ch=3):
        super().__init__()
        base = torchvision.models.mobilenet_v2(weights=None)
        base = adapt_first_conv(base, in_ch)

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(1280, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.head(x).squeeze(1)
        return x


@torch.no_grad()
def eval_metrics(model, loader, device):
    """
    Returns:
      acc: standard accuracy at threshold 0.5
      bal_acc: balanced accuracy = (TPR + TNR)/2
    Balanced accuracy is useful when labels are imbalanced.
    """
    model.eval()
    ys, ps = [], []

    for x, y in loader:
        x = x.to(device)
        prob = torch.sigmoid(model(x)).cpu().numpy()
        ys.extend(y.numpy().tolist())
        ps.extend(prob.tolist())

    ys = np.array(ys, dtype=int)
    ps = np.array(ps, dtype=float)
    pred = (ps >= 0.5).astype(int)

    acc = float((pred == ys).mean())

    # confusion components
    tp = int(((pred == 1) & (ys == 1)).sum())
    tn = int(((pred == 0) & (ys == 0)).sum())
    fp = int(((pred == 1) & (ys == 0)).sum())
    fn = int(((pred == 0) & (ys == 1)).sum())

    tpr = tp / max(1, int((ys == 1).sum()))
    tnr = tn / max(1, int((ys == 0).sum()))
    bal_acc = float(0.5 * (tpr + tnr))

    return acc, bal_acc, {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_root", default="data/work")
    ap.add_argument("--use_freq", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_csv = os.path.join(args.work_root, "splits", "train_frames.csv")
    dev_csv = os.path.join(args.work_root, "splits", "dev_frames.csv")

    ds_tr = FrameDS(train_csv, use_freq=args.use_freq)
    ds_de = FrameDS(dev_csv, use_freq=args.use_freq)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=2)
    dl_de = DataLoader(ds_de, batch_size=args.batch_size, shuffle=False, num_workers=2)

    in_ch = 4 if args.use_freq else 3
    model = MobileNetPAD(in_ch=in_ch).to(device)

    # class imbalance handling
    # live=1 is often fewer than attack=0 as per theory of the dataset
    train_labels = pd.read_csv(train_csv)["label"].values
    neg = int((train_labels == 0).sum())
    pos = int((train_labels == 1).sum())
    pos_weight = torch.tensor([neg / max(1, pos)], dtype=torch.float32, device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(
        "Using pos_weight:", float(pos_weight.item()), "| train neg:", neg, "pos:", pos
    )

    # optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    os.makedirs("runs", exist_ok=True)
    run_name = "msu_rgbfreq" if args.use_freq else "msu_rgb"
    best_path = f"runs/{run_name}_best.pt"

    # Used balanced accuracy to pick best model
    best_bal = -1.0

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []

        for x, y in tqdm(dl_tr, desc=f"epoch {ep}"):
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))

        dev_acc, dev_bal, cm = eval_metrics(model, dl_de, device)
        print(
            {
                "epoch": ep,
                "loss": float(np.mean(losses)),
                "dev_acc": dev_acc,
                "dev_bal_acc": dev_bal,
                "dev_confusion": cm,
            }
        )

        if dev_bal > best_bal:
            best_bal = dev_bal
            torch.save(
                {"model": model.state_dict(), "use_freq": args.use_freq}, best_path
            )

    print("Saved best checkpoint:", best_path)
    print("Best dev balanced accuracy:", best_bal)


if __name__ == "__main__":
    main()
