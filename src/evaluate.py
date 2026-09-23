"""학습된 모델을 Test 데이터로 평가.

실행: python src/evaluate.py --data data --ckpt outputs/resnet50_first/best.pt

만들어지는 것 (ckpt 와 같은 폴더):
    test_result.txt        Top-1, Top-3 정확도, 클래스별 정밀도·재현율, 악성 recall
    confusion_matrix.png   혼동행렬 그림 (미팅 때 보여줄 것)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import MALIGNANT, build_model, get_dataset, get_device, setup_korean_font


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in tqdm(loader, desc="평가"):
        out = model(x.to(device))
        probs.append(torch.softmax(out, 1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--ckpt", default="outputs/resnet50_first/best.pt")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--malignant-threshold", type=float, default=0.5,
                        help="악성 3종 확률 합이 이 값을 넘으면 '악성 의심'으로 판정")
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt["classes"]
    model = build_model(len(classes), ckpt["arch"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    ds = get_dataset(args.data, args.split, ckpt["img_size"])
    assert ds.classes == classes, f"클래스 순서가 다름:\n학습 {classes}\n평가 {ds.classes}"
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    probs, y_true = predict(model, loader, device)
    y_pred = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]

    top1 = (y_pred == y_true).mean()
    top3_acc = np.mean([y_true[i] in top3[i] for i in range(len(y_true))])

    # 2단계 출력: 악성 3종 확률 합 -> 악성 여부
    mal_idx = [classes.index(c) for c in MALIGNANT if c in classes]
    mal_prob = probs[:, mal_idx].sum(1)
    is_mal_true = np.isin(y_true, mal_idx)
    is_mal_pred = mal_prob >= args.malignant_threshold
    mal_recall = (is_mal_pred & is_mal_true).sum() / max(is_mal_true.sum(), 1)
    mal_precision = (is_mal_pred & is_mal_true).sum() / max(is_mal_pred.sum(), 1)
    benign_as_mal = (is_mal_pred & ~is_mal_true).sum() / max((~is_mal_true).sum(), 1)

    report = classification_report(y_true, y_pred, target_names=classes, digits=3)
    lines = [
        f"평가 split: {args.split}  ({len(ds)}장)",
        f"모델: {ckpt['arch']}  입력 크기: {ckpt['img_size']}",
        f"Top-1 정확도: {top1:.2%}",
        f"Top-3 정확도: {top3_acc:.2%}",
        "",
        f"[악성/양성 2단계 판정] 악성 3종 확률 합 >= {args.malignant_threshold}",
        f"  악성 recall (실제 악성 중 잡아낸 비율): {mal_recall:.2%}   <- 의료에서 가장 중요",
        f"  악성 precision: {mal_precision:.2%}",
        f"  양성을 악성으로 잘못 경고한 비율: {benign_as_mal:.2%}",
        "",
        "[클래스별 결과]",
        report,
    ]
    text = "\n".join(lines)
    print(text)
    out_dir = Path(args.ckpt).parent
    (out_dir / f"{args.split}_result.txt").write_text(text, encoding="utf-8")

    setup_korean_font()
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 11))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, xticks_rotation=60, colorbar=False)
    ax.set_title(f"혼동행렬 ({args.split}, Top-1 {top1:.2%})")
    ax.set_xlabel("모델이 예측한 종류"); ax.set_ylabel("실제 종류")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    print(f"\n저장: {out_dir / f'{args.split}_result.txt'}, {out_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
