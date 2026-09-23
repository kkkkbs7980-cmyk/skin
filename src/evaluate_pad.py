"""AI Hub 합성 이미지로 학습한 모델을 PAD-UFES-20 실제 사진으로 외부 평가한다.

실행 위치: project1
python "skin-tumor-screening_260914 (1)\src\evaluate_pad.py" --pad "zr7vgbcyr2-1" --ckpt "skin-tumor-screening_260914 (1)\outputs\resnet50_first\best.pt"
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import MALIGNANT, build_model, get_device, get_transforms, setup_korean_font


PAD_TO_AIHUB = {
    "ACK": "광선각화증",
    "BCC": "기저세포암",
    "MEL": "악성흑색종",
    "NEV": "멜라닌세포모반",
    "SCC": "편평세포암",
    "SEK": "지루각화증",
}


class PadDataset(Dataset):
    def __init__(self, rows, images, classes, img_size):
        self.rows = rows
        self.images = images
        self.labels = [classes.index(PAD_TO_AIHUB[row["diagnostic"]]) for row in rows]
        self.transform = get_transforms(img_size, train=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image_path = self.images[self.rows[index]["img_id"]]
        with Image.open(image_path) as image:
            x = self.transform(image.convert("RGB"))
        return x, self.labels[index]


def load_pad(pad_root):
    metadata_path = pad_root / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"PAD metadata.csv 없음: {metadata_path}")
    with metadata_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("PAD metadata.csv가 비어 있음")
    unknown = sorted({row["diagnostic"] for row in rows} - PAD_TO_AIHUB.keys())
    if unknown:
        raise ValueError(f"대응되지 않은 PAD 라벨: {unknown}")
    ids = [row["img_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("metadata.csv에 중복 img_id가 있음")

    images = {}
    for path in (pad_root / "images").rglob("*.png"):
        if path.name in images:
            raise ValueError(f"중복 이미지 파일명: {path.name}")
        images[path.name] = path
    missing = sorted(set(ids) - images.keys())
    if missing:
        raise FileNotFoundError(f"메타데이터의 이미지 {len(missing)}장 없음. 예: {missing[:5]}")
    return rows, images


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    chunks = []
    for x, _ in tqdm(loader, desc="PAD 외부 평가"):
        logits = model(x.to(device))
        chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(chunks)


def save_matrix(path, counts, row_names, col_names, normalized):
    setup_korean_font()
    shown = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1) if normalized else counts
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalized else None, aspect="auto")
    ax.set_xticks(range(len(col_names)), col_names, rotation=55, ha="right")
    ax.set_yticks(range(len(row_names)), row_names)
    ax.set_xlabel("모델의 15종 예측")
    ax.set_ylabel("PAD 실제 6종")
    ax.set_title("PAD-UFES-20 외부 평가 혼동행렬" + (" (행 정규화)" if normalized else " (건수)"))
    for row in range(len(row_names)):
        for col in range(len(col_names)):
            value = shown[row, col]
            label = f"{value:.0%}" if normalized else str(counts[row, col])
            ax.text(col, row, label, ha="center", va="center", fontsize=7,
                    color="white" if value > (0.5 if normalized else counts.max() / 2) else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pad", required=True, help="PAD-UFES-20 폴더 (metadata.csv와 images 포함)")
    parser.add_argument("--ckpt", required=True, help="AI Hub로만 학습한 best.pt")
    parser.add_argument("--out", default=None, help="결과 폴더 (기본: 체크포인트 옆 pad_external)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--malignant-threshold", type=float, default=0.5)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out) if args.out else ckpt_path.parent / "pad_external"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    classes = ckpt["classes"]
    missing_classes = sorted(set(PAD_TO_AIHUB.values()) - set(classes))
    if missing_classes:
        raise ValueError(f"모델에 PAD 대응 클래스가 없음: {missing_classes}")
    model = build_model(len(classes), ckpt["arch"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    rows, images = load_pad(Path(args.pad))
    ds = PadDataset(rows, images, classes, ckpt["img_size"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    probs = predict(model, loader, device)
    y_true = np.asarray(ds.labels)
    y_pred = probs.argmax(axis=1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    correct = y_pred == y_true
    top1_acc = correct.mean()
    top3_acc = np.mean([y_true[i] in top3[i] for i in range(len(y_true))])

    pad_codes = list(PAD_TO_AIHUB)
    pad_names = [PAD_TO_AIHUB[code] for code in pad_codes]
    pad_indices = [classes.index(name) for name in pad_names]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=pad_indices, zero_division=0)
    counts = np.zeros((len(pad_codes), len(classes)), dtype=int)
    row_lookup = {idx: row for row, idx in enumerate(pad_indices)}
    for true_idx, pred_idx in zip(y_true, y_pred):
        counts[row_lookup[true_idx], pred_idx] += 1

    malignant_indices = [classes.index(name) for name in MALIGNANT]
    mal_prob = probs[:, malignant_indices].sum(axis=1)
    true_mal = np.isin(y_true, malignant_indices)
    pred_mal = mal_prob >= args.malignant_threshold
    tp = int((pred_mal & true_mal).sum())
    fn = int((~pred_mal & true_mal).sum())
    fp = int((pred_mal & ~true_mal).sum())
    tn = int((~pred_mal & ~true_mal).sum())

    lines = [
        "PAD-UFES-20 실제 사진 외부 평가 (학습·모델 선택에 PAD를 사용하지 않은 경우)",
        f"평가 이미지: {len(rows)}장 / 모델: {ckpt['arch']} / 입력: {ckpt['img_size']}px",
        "모델은 15종 중 자유롭게 예측하며, PAD의 실제 라벨은 겹치는 6종입니다.",
        f"15종 직접 Top-1 정확도: {top1_acc:.2%} ({int(correct.sum())}/{len(rows)})",
        f"15종 직접 Top-3 정확도: {top3_acc:.2%}",
        f"겹치는 6종 Macro F1 (다른 9종 예측은 오답): {np.mean(f1):.3f}",
        "",
        "[클래스별 결과: 모델의 15종 직접 예측 기준]",
        "PAD  AI Hub 클래스       장수  precision  recall  F1",
    ]
    for code, name, n, p, r, f in zip(pad_codes, pad_names, support, precision, recall, f1):
        lines.append(f"{code:<3}  {name:<12} {n:>4}  {p:>9.3f}  {r:>6.3f}  {f:>5.3f}")
    lines += [
        "",
        f"[악성 3종 대 나머지, 확률 합 >= {args.malignant_threshold:g}]",
        f"실제 악성: {tp + fn}장 / 놓침: {fn}장 / 잘못 경고: {fp}장",
        f"악성 recall: {tp / (tp + fn):.2%}" if tp + fn else "악성 recall: 계산 불가",
        f"악성 precision: {tp / (tp + fp):.2%}" if tp + fp else "악성 precision: 계산 불가",
        f"비악성 잘못 경고율: {fp / (fp + tn):.2%}" if fp + tn else "비악성 잘못 경고율: 계산 불가",
        "주의: 위 확률 합은 보정된 임상 위험도가 아닙니다.",
        "주의: PAD는 환자당 여러 장이 있을 수 있으므로 이미지 단위 결과입니다.",
    ]
    report = "\n".join(lines) + "\n"
    print(report)
    (out_dir / "pad_external_result.txt").write_text(report, encoding="utf-8")
    save_matrix(out_dir / "pad_confusion_matrix_counts.png", counts, pad_names, classes, False)
    save_matrix(out_dir / "pad_confusion_matrix_normalized.png", counts, pad_names, classes, True)

    with (out_dir / "pad_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["img_id", "patient_id", "pad_label", "true_class", "predicted_class",
                         "correct", "malignant_score"])
        for row, true_idx, pred_idx, ok, score in zip(rows, y_true, y_pred, correct, mal_prob):
            writer.writerow([row["img_id"], row["patient_id"], row["diagnostic"],
                             classes[true_idx], classes[pred_idx], int(ok), f"{score:.6f}"])
    print(f"결과 저장: {out_dir}")


if __name__ == "__main__":
    main()
