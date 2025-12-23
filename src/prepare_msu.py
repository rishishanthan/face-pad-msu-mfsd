import argparse
import random
import re
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}


def norm_sid(s: str) -> str:
    """
    I Normalize subject id to 3-digit format.
    Like: "1"->"001", "001"->"001", "client001"->"001"
    """
    m = re.search(r"(\d+)", str(s))
    if not m:
        return ""
    return f"{int(m.group(1)):03d}"


def read_subject_list(p: Path) -> set:
    """
    Read train/test subject list file and normalize IDs.
    """
    ids = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        sid = norm_sid(line)
        if sid:
            ids.add(sid)
    return ids


def parse_video_name(name: str):
    """
    MSU naming pattern:
      real_clientID_cameraType_resolution_scene01.ext
      attack_clientID_cameraType_resolution_attackType_scene01.ext

    Returns: (kind, subject_id, camera, resolution, attack_type) or None
    """
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 5:
        return None

    kind = parts[0].lower()
    subject_id = norm_sid(parts[1])
    camera = parts[2]
    resolution = parts[3]
    attack_type = "live" if kind == "real" else parts[4].lower()

    if not subject_id:
        return None

    return kind, subject_id, camera, resolution, attack_type


def uniform_indices(n: int, k: int):
    """
    Select k frame indices uniformly across [0, n-1].
    """
    if n <= 0:
        return []
    if k <= 1:
        return [0]
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def extract_frames(video_path: Path, out_dir: Path, k: int):
    """
    Extract k frames per video and save as jpg.
    If decoding fails for some videos/frames, it may save fewer frames and continue.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = set(uniform_indices(n, k))

    out_dir.mkdir(parents=True, exist_ok=True)
    i = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if i in idxs:
            out_path = out_dir / f"frame_{i:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        i += 1

    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msu_root", required=True, help="Extracted MSU-MFSD folder")
    ap.add_argument(
        "--out_root",
        required=True,
        help="Output root (frames + splits will be created here)",
    )
    ap.add_argument("--frames_per_video", type=int, default=8)
    ap.add_argument(
        "--dev_frac",
        type=float,
        default=0.33,
        help="fraction of TRAIN subjects used as dev",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    msu_root = Path(args.msu_root)
    out_root = Path(args.out_root)

    frames_root = out_root / "frames"
    splits_root = out_root / "splits"
    frames_root.mkdir(parents=True, exist_ok=True)
    splits_root.mkdir(parents=True, exist_ok=True)

    train_ids = read_subject_list(msu_root / "train_sub_list.txt")
    test_ids = read_subject_list(msu_root / "test_sub_list.txt")

    real_dir = msu_root / "scene01" / "real"
    atk_dir = msu_root / "scene01" / "attack"

    videos = []

    # label: 1 = live, 0 = attack
    for base, label in [(real_dir, 1), (atk_dir, 0)]:
        for vp in base.rglob("*"):
            if vp.suffix.lower() not in VIDEO_EXTS:
                continue

            meta = parse_video_name(vp.name)
            if meta is None:
                continue

            _, sid, cam, res, atk = meta
            videos.append(
                {
                    "video_path": str(vp),
                    "video_name": vp.name,
                    "video_id": vp.stem,
                    "subject_id": sid,
                    "label": int(label),
                    "attack_type": atk,
                    "camera": cam,
                    "resolution": res,
                }
            )

    dfv = pd.DataFrame(videos)
    if dfv.empty:
        raise RuntimeError("No videos found. Check msu_root path and folder structure.")

    def split_of(sid: str) -> str:
        if sid in test_ids:
            return "test"
        if sid in train_ids:
            return "train"
        return "train"  # fallback

    dfv["split"] = dfv["subject_id"].apply(split_of)

    train_subjects = sorted(
        dfv[dfv["split"] == "train"]["subject_id"].unique().tolist()
    )
    random.shuffle(train_subjects)

    n_dev = max(1, int(args.dev_frac * len(train_subjects)))
    dev_subjects = set(train_subjects[:n_dev])

    dfv.loc[
        (dfv["split"] == "train") & (dfv["subject_id"].isin(dev_subjects)), "split"
    ] = "dev"

    # Saved video-level CSVs here for the graph
    for s in ["train", "dev", "test"]:
        dfv[dfv["split"] == s].to_csv(splits_root / f"{s}_videos.csv", index=False)

    # Extract frames and build frame-level CSVs
    frame_rows = {"train": [], "dev": [], "test": []}

    for _, r in tqdm(dfv.iterrows(), total=len(dfv), desc="Extracting frames"):
        split = r["split"]
        vid = r["video_id"]
        vp = Path(r["video_path"])

        out_dir = frames_root / split / vid
        saved = extract_frames(vp, out_dir, args.frames_per_video)

        if saved == 0:
            continue

        for fp in sorted(out_dir.glob("frame_*.jpg")):
            frame_rows[split].append(
                {
                    "frame_path": str(fp),
                    "label": int(r["label"]),
                    "video_id": vid,
                    "subject_id": r["subject_id"],
                    "attack_type": r["attack_type"],
                }
            )

    for s in ["train", "dev", "test"]:
        pd.DataFrame(frame_rows[s]).to_csv(splits_root / f"{s}_frames.csv", index=False)

    print("DONE")
    print("Splits:", splits_root)
    print("Frames:", frames_root)
    print("Video counts:", dfv["split"].value_counts().to_dict())
    print(
        "Unique subjects by split:",
        dfv.groupby("split")["subject_id"].nunique().to_dict(),
    )
    print("Dev fraction (subjects):", args.dev_frac)
    print("Frames per video:", args.frames_per_video)


if __name__ == "__main__":
    main()
