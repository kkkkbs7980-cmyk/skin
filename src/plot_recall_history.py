"""균등한 AI Hub 15종 데이터의 epoch별 Macro Recall 곡선을 그린다.

각 클래스 장수가 같을 때 Macro Recall은 정확도와 정확히 같다.
history.csv의 train_acc/val_acc를 재사용하며 모델을 다시 학습하지 않는다.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import get_dataset, setup_korean_font


def check_balanced(data_root, split):
    dataset = get_dataset(data_root, split, img_size=256)
    counts = Counter(dataset.targets)
    if len(counts) != len(dataset.classes) or len(set(counts.values())) != 1:
        raise ValueError(f"{split} 클래스별 장수가 달라 정확도로 Macro Recall을 복원할 수 없습니다: {counts}")
    return len(dataset.classes), next(iter(counts.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="AI Hub 데이터 폴더")
    parser.add_argument("--history", required=True, help="train.py가 저장한 history.csv")
    args = parser.parse_args()

    train_classes, train_per_class = check_balanced(args.data, "train")
    val_classes, val_per_class = check_balanced(args.data, "val")
    if train_classes != val_classes:
        raise ValueError("학습과 검증 클래스 수가 다릅니다")

    history_path = Path(args.history)
    with history_path.open(newline="", encoding="utf-8") as f:
        history = list(csv.DictReader(f))
    if not history:
        raise ValueError("history.csv가 비어 있습니다")
    epochs = [int(row["epoch"]) for row in history]
    train_recall = [float(row["train_acc"]) for row in history]
    val_recall = [float(row["val_acc"]) for row in history]
    if any(not 0 <= value <= 1 for value in train_recall + val_recall):
        raise ValueError("정확도 값이 0~1 범위를 벗어났습니다")

    out_dir = history_path.parent
    csv_path = out_dir / "macro_recall_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_macro_recall", "val_macro_recall"])
        writer.writerows(zip(epochs, train_recall, val_recall))

    setup_korean_font()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [x * 100 for x in train_recall], marker="o", label="학습 Macro Recall")
    ax.plot(epochs, [x * 100 for x in val_recall], marker="s", label="검증 Macro Recall")
    ax.set_xticks(epochs)
    ax.set_ylim(90, 100.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro Recall (%)")
    ax.set_title("AI Hub 15종 Macro Recall 학습곡선")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.text(0.5, 0.01,
             f"각 클래스 동일 장수: 학습 {train_per_class}장, 검증 {val_per_class}장 → Macro Recall = 정확도",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    png_path = out_dir / "macro_recall_curve.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"저장: {png_path}")
    print(f"저장: {csv_path}")


if __name__ == "__main__":
    main()
