"""첫 학습: 미리 학습된 ResNet50을 우리 데이터 15종으로 다시 학습 (전이학습).

실행 (GPU 데스크탑, conda 환경에서):
    python src/train.py --data data --epochs 10

끝나면 outputs/<이름>/ 에 다음이 생긴다.
    best.pt        검증 정확도가 가장 좋았던 모델 (evaluate.py 에서 사용)
    history.csv    epoch별 손실·정확도
    history.png    학습 곡선 그림
    classes.txt    클래스 이름 순서
"""
import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import build_model, get_dataset, get_device, setup_korean_font


def run_one_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
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
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="outputs/resnet50_first", help="결과 저장 폴더")
    parser.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet101", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32, help="8GB GPU에서 메모리 부족이면 16으로")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4, help="데이터 읽기 프로세스 수. 문제 생기면 0")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    torch.manual_seed(0)

    train_ds = get_dataset(args.data, "train", args.img_size)
    val_ds = get_dataset(args.data, "val", args.img_size)
    classes = train_ds.classes
    print(f"클래스 {len(classes)}개: {classes}")
    print(f"학습 {len(train_ds)}장, 검증 {len(val_ds)}장, 장치 {device}")
    (out_dir / "classes.txt").write_text("\n".join(classes), encoding="utf-8")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = build_model(len(classes), args.arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        history.append([epoch, tr_loss, tr_acc, va_loss, va_acc])
        print(f"[{epoch}/{args.epochs}] 학습 loss {tr_loss:.3f} acc {tr_acc:.3%} | "
              f"검증 loss {va_loss:.3f} acc {va_acc:.3%} | {time.time() - t0:.0f}초")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(), "classes": classes,
                        "arch": args.arch, "img_size": args.img_size}, out_dir / "best.pt")
            print(f"  -> best.pt 저장 (검증 acc {best_acc:.3%})")

    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        w.writerows(history)

    # 학습 곡선 그림
    setup_korean_font()
    import matplotlib.pyplot as plt
    ep = [h[0] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, [h[1] for h in history], label="학습")
    ax[0].plot(ep, [h[3] for h in history], label="검증")
    ax[0].set_title("손실(loss)"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [h[2] for h in history], label="학습")
    ax[1].plot(ep, [h[4] for h in history], label="검증")
    ax[1].set_title("정확도"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "history.png", dpi=120)

    print(f"\n완료. 최고 검증 정확도 {best_acc:.2%}. 결과 폴더: {out_dir}")
    print(f"다음: python src/evaluate.py --data {args.data} --ckpt {out_dir / 'best.pt'} --split val")


if __name__ == "__main__":
    main()
