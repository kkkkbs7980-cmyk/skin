"""학습된 모델을 test 로 평가.

실행: python src/evaluate.py --data data --ckpt outputs/A_baseline/best.pt

만들어지는 것 (ckpt 와 같은 폴더):
    test_result.txt        정확도, balanced accuracy, macro F1, 클래스별 recall, 악성 위험도 판정
    test_summary.json      compare.py 가 실험끼리 표로 묶을 때 쓰는 숫자
    confusion_matrix.png   혼동행렬 그림 (미팅 때 보여줄 것)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import CLASS_KO, CLASSES, MALIGNANT, HamDataset, build_model, get_device, setup_korean_font


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in tqdm(loader, desc="평가"):
        out = model(x.to(device))
        probs.append(torch.softmax(out, 1).cpu())
        labels.append(torch.as_tensor(y))
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--ckpt", default="outputs/A_baseline/best.pt")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--malignant-threshold", type=float, default=0.5,
                        help="악성·전암 3종(mel, bcc, akiec) 확률 합이 이 값을 넘으면 '전문의 상담 권고'")
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    assert ckpt["classes"] == CLASSES
    model = build_model(len(CLASSES), ckpt["arch"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    ds = HamDataset(args.data, args.split, ckpt["img_size"], aug=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    probs, y_true = predict(model, loader, device)
    y_pred = probs.argmax(1)

    acc = (y_pred == y_true).mean()
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    recalls = recall_score(y_true, y_pred, average=None, labels=range(len(CLASSES)))
    majority = np.bincount(y_true).max() / len(y_true)

    # 2단계 출력: 악성·전암 3종 확률 합 -> 전문의 상담 권고 여부
    mal_idx = [CLASSES.index(c) for c in MALIGNANT]
    mal_prob = probs[:, mal_idx].sum(1)
    is_mal_true = np.isin(y_true, mal_idx)
    is_mal_pred = mal_prob >= args.malignant_threshold
    mal_recall = (is_mal_pred & is_mal_true).sum() / max(is_mal_true.sum(), 1)
    mal_precision = (is_mal_pred & is_mal_true).sum() / max(is_mal_pred.sum(), 1)
    benign_as_mal = (is_mal_pred & ~is_mal_true).sum() / max((~is_mal_true).sum(), 1)

    names = [f"{c} {CLASS_KO[c]}" for c in CLASSES]
    report = classification_report(y_true, y_pred, labels=range(len(CLASSES)), target_names=names, digits=3)
    exp_name = Path(args.ckpt).parent.name
    lines = [
        f"실험: {exp_name}   (증강 {'켬' if ckpt.get('aug') else '끔'}, 불균형 보정 {ckpt.get('balance', 'none')}, "
        f"best epoch {ckpt.get('epoch', '?')})",
        f"평가 split: {args.split}  ({len(ds)}장)   모델: {ckpt['arch']}  입력 크기: {ckpt['img_size']}",
        "",
        f"정확도(accuracy):          {acc:.2%}   (전부 nv 로 찍으면 {majority:.1%})",
        f"balanced accuracy:         {bal_acc:.2%}   <- 클래스별 recall 의 평균. 불균형 데이터에서 진짜 실력",
        f"macro F1:                  {macro_f1:.3f}",
        f"악성흑색종(mel) recall:    {recalls[CLASSES.index('mel')]:.2%}   <- 놓치면 안 되는 것",
        "",
        f"[악성 위험도 2단계 판정] mel+bcc+akiec 확률 합 >= {args.malignant_threshold} 이면 '전문의 상담 권고'",
        f"  악성·전암 recall (실제 악성·전암 중 잡아낸 비율): {mal_recall:.2%}",
        f"  악성·전암 precision: {mal_precision:.2%}",
        f"  양성을 악성으로 잘못 경고한 비율: {benign_as_mal:.2%}",
        "",
        "[클래스별 결과]",
        report,
    ]
    text = "\n".join(lines)
    print(text)
    out_dir = Path(args.ckpt).parent
    (out_dir / f"{args.split}_result.txt").write_text(text, encoding="utf-8")

    summary = {
        "experiment": exp_name, "split": args.split, "n": int(len(ds)),
        "aug": bool(ckpt.get("aug")), "balance": ckpt.get("balance", "none"),
        "accuracy": float(acc), "balanced_accuracy": float(bal_acc), "macro_f1": float(macro_f1),
        "recall": {c: float(r) for c, r in zip(CLASSES, recalls)},
        "malignant_recall": float(mal_recall), "malignant_precision": float(mal_precision),
        "benign_false_alarm": float(benign_as_mal),
    }
    (out_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")

    setup_korean_font()
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"혼동행렬 ({exp_name}, {args.split})  정확도 {acc:.1%}  balanced acc {bal_acc:.1%}")
    ax.set_xlabel("모델이 예측한 종류"); ax.set_ylabel("실제 종류")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    print(f"\n저장: {out_dir / f'{args.split}_result.txt'}, {out_dir / 'confusion_matrix.png'}")
    print("실험 여러 개를 표로 묶으려면: python src/compare.py")


if __name__ == "__main__":
    main()
