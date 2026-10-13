"""HAM10000 학습: 미리 학습된 ResNet50 을 7종으로 다시 학습 (전이학습).

미팅용 실험 3개 (분할·모델·epoch 는 같고 조건만 다름):
    A 원본:              python src/train.py --data data --out outputs/A_baseline
    B 증강:              python src/train.py --data data --out outputs/B_aug --aug
    C 증강+불균형 보정:   python src/train.py --data data --out outputs/C_aug_weight --aug --balance weight

끝나면 outputs/<이름>/ 에 다음이 생긴다.
    best.pt        검증 balanced accuracy 가 가장 좋았던 모델 (evaluate.py 에서 사용)
    history.csv    epoch별 손실·정확도·balanced accuracy
    history.png    학습 곡선 그림
    config.txt     이 실험의 설정 (보고서에 그대로 쓰면 됨)
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from common import CLASSES, HamDataset, build_model, get_device, setup_korean_font


def run_one_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss, n = 0.0, 0
    preds, labels = [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, desc="학습" if train else "검증", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            preds.append(out.argmax(1).cpu())
            labels.append(y.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = (preds == labels).mean()
    bal_acc = balanced_accuracy_score(labels, preds)
    return total_loss / n, acc, bal_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="outputs/A_baseline", help="결과 저장 폴더")
    parser.add_argument("--aug", action="store_true", help="데이터 증강 켜기 (뒤집기·회전·색 변화·자르기)")
    parser.add_argument("--balance", default="none", choices=["none", "weight", "sampler"],
                        help="불균형 보정. weight: 드문 클래스의 loss 를 크게. sampler: 드문 클래스를 더 자주 뽑기")
    parser.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet18", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32, help="8GB GPU 에서 메모리 부족이면 16")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4, help="데이터 읽기 프로세스 수. 문제 생기면 0")
    parser.add_argument("--no-pretrained", action="store_true", help="(테스트용) ImageNet 가중치 안 씀")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    torch.manual_seed(0)
    np.random.seed(0)

    train_ds = HamDataset(args.data, "train", args.img_size, aug=args.aug)
    val_ds = HamDataset(args.data, "val", args.img_size, aug=False)
    counts = train_ds.class_counts()
    print(f"클래스 {len(CLASSES)}개: {CLASSES}")
    print(f"학습 {len(train_ds)}장 {dict(zip(CLASSES, counts))}")
    print(f"검증 {len(val_ds)}장, 장치 {device}, 증강 {'켬' if args.aug else '끔'}, 불균형 보정 {args.balance}")

    # 불균형 보정: 클래스 빈도의 역수를 가중치로 (드문 클래스 = 큰 가중치)
    inv = 1.0 / np.array(counts, dtype=np.float64)
    class_w = torch.tensor(inv / inv.sum() * len(CLASSES), dtype=torch.float32)

    sampler = None
    shuffle = True
    if args.balance == "sampler":
        sample_w = [inv[t] for t in train_ds.targets]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(train_ds), replacement=True)
        shuffle = False
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = build_model(len(CLASSES), args.arch, pretrained=not args.no_pretrained).to(device)
    if args.balance == "weight":
        criterion = nn.CrossEntropyLoss(weight=class_w.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    config = "\n".join(f"{k}: {v}" for k, v in vars(args).items())
    (out_dir / "config.txt").write_text(config + f"\n클래스 가중치: {[round(float(w), 2) for w in class_w]}\n",
                                        encoding="utf-8")

    history = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_bal = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc, va_bal = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        history.append([epoch, tr_loss, tr_acc, tr_bal, va_loss, va_acc, va_bal])
        print(f"[{epoch}/{args.epochs}] 학습 loss {tr_loss:.3f} acc {tr_acc:.1%} | "
              f"검증 loss {va_loss:.3f} acc {va_acc:.1%} bal.acc {va_bal:.1%} | {time.time() - t0:.0f}초")

        # 모델 선택 기준은 정확도가 아니라 balanced accuracy (nv 쏠림 방지)
        if va_bal > best:
            best = va_bal
            torch.save({"model": model.state_dict(), "classes": CLASSES, "arch": args.arch,
                        "img_size": args.img_size, "aug": args.aug, "balance": args.balance,
                        "epoch": epoch}, out_dir / "best.pt")
            print(f"  -> best.pt 저장 (검증 balanced acc {best:.1%})")

    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "train_bal_acc", "val_loss", "val_acc", "val_bal_acc"])
        w.writerows(history)

    setup_korean_font()
    import matplotlib.pyplot as plt
    ep = [h[0] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, [h[1] for h in history], color="#2a78d6", label="학습")
    ax[0].plot(ep, [h[4] for h in history], color="#eb6834", label="검증")
    ax[0].set_title("손실(loss)"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [h[5] for h in history], color="#eb6834", label="검증 정확도")
    ax[1].plot(ep, [h[6] for h in history], color="#1baf7a", label="검증 balanced acc")
    ax[1].set_title("검증 성능"); ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1); ax[1].legend()
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "history.png", dpi=120)

    print(f"\n완료. 최고 검증 balanced accuracy {best:.1%}. 결과 폴더: {out_dir}")
    print(f"다음: python src/evaluate.py --data {args.data} --ckpt {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
