"""학습된 3분류 모델을 test 로 평가.

실행: python src/evaluate.py --data data --ckpt outputs_3cls_v3/00_baseline/best.pt

만들어지는 것 (ckpt 와 같은 폴더):
    test_result.txt        결과표(행=클래스, 열=지표) + 참고 지표 + 악성 위험도 판정
    test_summary.json      클래스별 recall/precision/F1/AUC/accuracy(one-vs-rest), macro 평균, 전체 accuracy
                           (compare.py 가 결과표.md 만들 때 씀)
    confusion_matrix.png   혼동행렬 그림 (보고서용)

표 괄호 안 숫자 = 같은 폴더의 원본(예: outputs_3cls_v3/00_baseline) 대비 증가폭(%p). 원본 결과가 아직 없으면 괄호 없음.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (GROUP_KO, GROUPS, TABLE_NOTES, HamDataset, build_model, get_device, result_table,
                    setup_korean_font)

BASELINE_NAME = "00_baseline"


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in tqdm(loader, desc="평가"):
        out = model(x.to(device))
        probs.append(torch.softmax(out, 1).cpu())
        labels.append(torch.as_tensor(y))
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def load_baseline(exp_dir, split):
    """같은 결과 폴더(outputs_3cls_v3 등) 안의 00_baseline 결과. 없으면 None."""
    f = Path(exp_dir).parent / BASELINE_NAME / f"{split}_summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--ckpt", default="outputs_3cls_v3/00_baseline/best.pt")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--malignant-threshold", type=float, default=0.5,
                        help="악성 확률이 이 값 이상이면 '전문의 상담 권고' (참고용, 표에는 안 넣음)")
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if ckpt["classes"] != GROUPS:
        raise ValueError(f"이 모델은 3분류 모델이 아님 (classes={ckpt['classes']})")
    model = build_model(len(GROUPS), ckpt["arch"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    ds = HamDataset(args.data, args.split, ckpt["img_size"], aug=())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    probs, y_true = predict(model, loader, device)
    y_pred = probs.argmax(1)
    labels = list(range(len(GROUPS)))

    acc = float((y_pred == y_true).mean())
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    auc = []
    for k in labels:  # 클래스별 one-vs-rest AUC (softmax 확률)
        pos = (y_true == k)
        auc.append(float(roc_auc_score(pos, probs[:, k])) if 0 < pos.sum() < len(pos) else float("nan"))
    # 클래스별 accuracy = "그 클래스인가 아닌가" 2진 정확도 (TP+TN)/전체. AUC 열과 같은 one-vs-rest 방식
    class_acc = [float(((y_true == k) == (y_pred == k)).mean()) for k in labels]
    macro = {"recall": float(np.mean(rec)), "precision": float(np.mean(prec)), "f1": float(np.mean(f1)),
             "auc": float(np.nanmean(auc))}

    # 2단계 출력(참고): 악성 확률 >= 임계값 → 전문의 상담 권고
    mal = GROUPS.index("malignant")
    is_mal_true = y_true == mal
    is_mal_pred = probs[:, mal] >= args.malignant_threshold
    mal_recall = (is_mal_pred & is_mal_true).sum() / max(is_mal_true.sum(), 1)
    false_alarm = (is_mal_pred & ~is_mal_true).sum() / max((~is_mal_true).sum(), 1)

    exp_dir = Path(args.ckpt).parent
    exp_name = exp_dir.name
    summary = {
        "experiment": exp_name, "split": args.split, "n": int(len(ds)),
        "aug": ckpt.get("aug", []), "aug_desc": ckpt.get("aug_desc", ""), "balance": ckpt.get("balance", "none"),
        "arch": ckpt.get("arch"), "train_desc": ckpt.get("train_desc", ""),
        "epochs": ckpt.get("epochs", "?"), "best_epoch": ckpt.get("epoch", "?"),
        "recall": {g: float(v) for g, v in zip(GROUPS, rec)},
        "precision": {g: float(v) for g, v in zip(GROUPS, prec)},
        "f1": {g: float(v) for g, v in zip(GROUPS, f1)},
        "auc": dict(zip(GROUPS, auc)),
        "class_accuracy": dict(zip(GROUPS, class_acc)),
        "macro": macro,
        "accuracy": acc, "balanced_accuracy": bal_acc,
        "malignant_alarm_recall": float(mal_recall), "nonmalignant_false_alarm": float(false_alarm),
    }

    table = result_table(summary, load_baseline(exp_dir, args.split))
    counts = np.bincount(y_true, minlength=len(GROUPS))
    lines = [
        f"실험: {exp_name}   평가 split: {args.split} ({len(ds)}장: "
        + ", ".join(f"{GROUP_KO[g]} {c}" for g, c in zip(GROUPS, counts)) + ")",
        "",
        table,
        "",
        *TABLE_NOTES,
        "",
        f"(참고) balanced accuracy: {bal_acc:.2%}",
        f"(참고) 악성 확률 >= {args.malignant_threshold} → '전문의 상담 권고' 판정 시: "
        f"실제 악성 중 권고 {mal_recall:.1%}, 악성 아닌데 권고 {false_alarm:.1%}",
    ]
    text = "\n".join(lines)
    print(text)
    (exp_dir / f"{args.split}_result.txt").write_text(text, encoding="utf-8")
    (exp_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")

    setup_korean_font()
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=[GROUP_KO[g] for g in GROUPS]).plot(ax=ax, colorbar=False,
                                                                                 cmap="Blues")
    ax.set_title(f"혼동행렬 ({exp_name}, {args.split})\n정확도 {acc:.1%}  balanced acc {bal_acc:.1%}")
    ax.set_xlabel("모델이 예측한 종류"); ax.set_ylabel("실제 종류")
    fig.tight_layout()
    fig.savefig(exp_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    print(f"\n저장: {exp_dir / f'{args.split}_result.txt'}, {exp_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
