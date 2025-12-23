import argparse, os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision


def laplacian_channel(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap = lap - lap.min()
    if lap.max() > 1e-6:
        lap = lap / lap.max()
    lap = (lap * 255.0).astype(np.uint8)
    return lap


class FrameDS(Dataset):
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
            x = np.concatenate([rgb, lap[..., None]], axis=2)
        else:
            x = rgb

        x = torch.from_numpy(x).float() / 255.0
        x = x.permute(2, 0, 1).contiguous()
        y = int(r["label"])  # 1 live, 0 attack
        vid = r["video_id"]
        return x, y, vid


def adapt_first_conv(m, in_ch):
    conv = m.features[0][0]
    if conv.in_channels == in_ch:
        return m
    new = nn.Conv2d(
        in_ch,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        bias=False,
    )
    with torch.no_grad():
        w = conv.weight
        mean = w.mean(dim=1, keepdim=True)
        new.weight.copy_(torch.cat([w, mean.repeat(1, in_ch - 3, 1, 1)], dim=1))
    m.features[0][0] = new
    return m


class MobileNetPAD(nn.Module):
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
        return self.head(x).squeeze(1)


@torch.no_grad()
def frame_probs(model, loader, device):
    model.eval()
    rows = []
    for x, y, vid in tqdm(loader, desc="Scoring frames"):
        x = x.to(device)
        p = torch.sigmoid(model(x)).cpu().numpy()
        for i in range(len(p)):
            rows.append((vid[i], int(y[i]), float(p[i])))
    return pd.DataFrame(rows, columns=["video_id", "label", "prob_live"])


def video_aggregate(df_frame):
    return df_frame.groupby(["video_id", "label"])["prob_live"].mean().reset_index()


def rates(scores, labels, thr):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred_live = (scores >= thr).astype(int)

    apcer = ((pred_live == 1) & (labels == 0)).sum() / max(
        1, (labels == 0).sum()
    )  # attacks -> live
    bpcer = ((pred_live == 0) & (labels == 1)).sum() / max(
        1, (labels == 1).sum()
    )  # live -> attack
    acer = (apcer + bpcer) / 2.0
    acc = (pred_live == labels).mean()
    return float(apcer), float(bpcer), float(acer), float(acc)


def find_eer_threshold(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    thr_list = np.unique(scores)

    best = None
    for t in thr_list:
        apcer, bpcer, _, _ = rates(scores, labels, t)
        diff = abs(apcer - bpcer)
        if best is None or diff < best[0]:
            best = (diff, t, apcer, bpcer)
    _, t, ap, bp = best
    eer = (ap + bp) / 2.0
    return float(t), float(eer)


def find_min_acer_threshold(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    thr_list = np.unique(scores)

    best = None
    for t in thr_list:
        apcer, bpcer, acer, acc = rates(scores, labels, t)
        if best is None or acer < best[0]:
            best = (acer, t, apcer, bpcer, acc)
    acer, t, ap, bp, acc = best
    return float(t), float(acer), float(ap), float(bp), float(acc)


def find_apcer_constrained_threshold(scores, labels, max_apcer=0.01):
    # I chosen the lowest BPCER while keeping APCER <= max_apcer
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    thr_list = np.unique(scores)

    best = None
    for t in thr_list:
        apcer, bpcer, acer, acc = rates(scores, labels, t)
        if apcer <= max_apcer:
            if best is None or bpcer < best[0]:
                best = (bpcer, t, apcer, acer, acc)
    return best  # may be None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_root", default="data/work")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_apcer", type=float, default=0.01)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu")
    use_freq = bool(ck.get("use_freq", False))
    in_ch = 4 if use_freq else 3

    model = MobileNetPAD(in_ch=in_ch).to(device)
    model.load_state_dict(ck["model"], strict=True)

    dev_csv = os.path.join(args.work_root, "splits", "dev_frames.csv")
    test_csv = os.path.join(args.work_root, "splits", "test_frames.csv")

    dl_dev = DataLoader(
        FrameDS(dev_csv, use_freq=use_freq),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    dl_test = DataLoader(
        FrameDS(test_csv, use_freq=use_freq),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    df_dev_frames = frame_probs(model, dl_dev, device)
    df_test_frames = frame_probs(model, dl_test, device)

    dev_vid = video_aggregate(df_dev_frames)
    test_vid = video_aggregate(df_test_frames)

    # I am saving video-level scores so I did the plot later in notebook
    os.makedirs("runs", exist_ok=True)
    dev_vid.to_csv("runs/dev_video_scores.csv", index=False)
    test_vid.to_csv("runs/test_video_scores.csv", index=False)

    # 1) EER threshold (dev)
    thr_eer, dev_eer = find_eer_threshold(
        dev_vid["prob_live"].values, dev_vid["label"].values
    )
    ap, bp, acer, acc = rates(
        test_vid["prob_live"].values, test_vid["label"].values, thr_eer
    )

    print("=== VIDEO LEVEL RESULTS ===")
    print({"use_freq": use_freq, "dev_thr_eer": thr_eer, "dev_eer": dev_eer})
    print({"test_apcer": ap, "test_bpcer": bp, "test_acer": acer, "test_acc": acc})

    # 2) Min-ACER threshold (dev)
    thr_min, dev_acer, dev_ap, dev_bp, dev_acc = find_min_acer_threshold(
        dev_vid["prob_live"].values, dev_vid["label"].values
    )
    ap2, bp2, acer2, acc2 = rates(
        test_vid["prob_live"].values, test_vid["label"].values, thr_min
    )
    print(
        {
            "dev_thr_min_acer": thr_min,
            "dev_acer": dev_acer,
            "dev_apcer": dev_ap,
            "dev_bpcer": dev_bp,
            "dev_acc": dev_acc,
        }
    )
    print(
        {
            "test_apcer_at_min_acer": ap2,
            "test_bpcer_at_min_acer": bp2,
            "test_acer_at_min_acer": acer2,
            "test_acc_at_min_acer": acc2,
        }
    )

    # 3) Low-APCER operating point (dev),
    best = find_apcer_constrained_threshold(
        dev_vid["prob_live"].values, dev_vid["label"].values, max_apcer=args.max_apcer
    )
    if best is None:
        print(
            {
                "dev_apcer_constraint": args.max_apcer,
                "status": "No threshold meets APCER constraint on dev",
            }
        )
    else:
        bp3, thr3, ap3, acer3, acc3 = best
        ap_t, bp_t, acer_t, acc_t = rates(
            test_vid["prob_live"].values, test_vid["label"].values, thr3
        )
        print(
            {
                "dev_apcer_constraint": args.max_apcer,
                "dev_thr": float(thr3),
                "dev_apcer": float(ap3),
                "dev_bpcer": float(bp3),
            }
        )
        print(
            {
                "test_apcer_at_constraint": ap_t,
                "test_bpcer_at_constraint": bp_t,
                "test_acer_at_constraint": acer_t,
                "test_acc_at_constraint": acc_t,
            }
        )


if __name__ == "__main__":
    main()
