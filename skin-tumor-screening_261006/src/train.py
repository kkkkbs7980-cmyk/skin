"""HAM10000 3분류(정상/양성/악성) 학습: 미리 학습된 ResNet50 을 다시 학습 (전이학습).

실행 예:
    원본:            python src/train.py --data data --out outputs_3cls/00_baseline
    증강 하나:       python src/train.py --data data --out outputs_3cls/01_flip --aug flip
    증강 여러 개:    python src/train.py --data data --out outputs_3cls/05_combo --aug flip crop cutmix
    가능한 증강 이름: flip crop color cutmix hair vignette erase elastic   (전체 순서는 run_all.bat)

끝나면 outputs_3cls/<이름>/ 에 다음이 생긴다.
    best.pt        검증 balanced accuracy 가 가장 좋았던 모델 (evaluate.py 에서 사용)
    history.csv    epoch별 손실·정확도·balanced accuracy, 마지막 줄에 총 학습 시간(분)
    history.png    학습 곡선 그림
    config.txt     첫 줄: 사용한 증강 설정 (결과표 위에 그대로 실림), 그 아래 세부 설정과 총 학습 시간
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

from common import AUG_NAMES, GROUPS, HamDataset, aug_description, build_model, get_device, setup_korean_font


def cutmix(x, y, alpha=1.0):
    """CutMix: 배치 안에서 다른 사진의 사각형 조각을 잘라 붙인다. (섞인 x, 원래 y, 짝 y, λ) 반환.
    λ ~ Beta(alpha, alpha) 로 뽑고, 실제로 붙인 넓이에 맞게 다시 계산한다. 손실은 λ 비율로 섞는다."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    h, w = x.shape[2], x.shape[3]
    cut = np.sqrt(1.0 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    return x, y, y[idx], lam


def run_one_epoch(model, loader, criterion, optimizer, device, train, use_cutmix=False):
    model.train(train)
    total_loss, n = 0.0, 0
    preds, labels = [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, desc="학습" if train else "검증", leave=False):
            x, y = x.to(device), y.to(device)
            if train and use_cutmix:
                x, ya, yb, lam = cutmix(x, y)
                out = model(x)
                loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
            else:
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            preds.append(out.argmax(1).cpu())
            labels.append(y.cpu())  # CutMix 때도 원래 라벨 기준 (학습 정확도는 대략치)
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = (preds == labels).mean()
    bal_acc = balanced_accuracy_score(labels, preds)
    return total_loss / n, acc, bal_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="outputs_3cls/00_baseline", help="결과 저장 폴더")
    parser.add_argument("--aug", nargs="*", default=[], choices=AUG_NAMES,
                        help="증강 기법 이름 여러 개. 예: --aug flip crop cutmix. 없으면 원본")
    parser.add_argument("--balance", default="none", choices=["none", "weight", "sampler"],
                        help="불균형 보정. weight: 드문 클래스의 loss 를 크게. sampler: 드문 클래스를 더 자주 뽑기")
    parser.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet18", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32, help="8GB GPU 에서 메모리 부족이면 16")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4, help="데이터 읽기 프로세스 수. 문제 생기면 0")
    parser.add_argument("--no-pretrained", action="store_true", help="(테스트용) ImageNet 가중치 안 씀")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 이전에 같은 폴더에서 돌다 만 결과가 남아 있으면 결과표에 섞이지 않게 지운다
    for old in ("best.pt", "test_summary.json", "test_result.txt"):
        (out_dir / old).unlink(missing_ok=True)
    device = get_device()
    torch.manual_seed(0)
    np.random.seed(0)

    aug = list(dict.fromkeys(args.aug))  # 중복 제거, 순서 유지
    aug_desc = aug_description(aug)
    use_cutmix = "cutmix" in aug
    train_ds = HamDataset(args.data, "train", args.img_size, aug=aug)
    val_ds = HamDataset(args.data, "val", args.img_size, aug=())
    counts = train_ds.class_counts()
    print(f"클래스 {len(GROUPS)}개: {GROUPS}")
    print(f"학습 {len(train_ds)}장 {dict(zip(GROUPS, counts))}")
    print(f"검증 {len(val_ds)}장, 장치 {device}, 불균형 보정 {args.balance}")
    print(f"설정: {aug_desc}")

    # 불균형 보정: 클래스 빈도의 역수를 가중치로 (드문 클래스 = 큰 가중치)
    inv = 1.0 / np.maximum(np.array(counts, dtype=np.float64), 1)
    class_w = torch.tensor(inv / inv.sum() * len(GROUPS), dtype=torch.float32)

    sampler = None
    shuffle = True
    if args.balance == "sampler":
        sample_w = [inv[t] for t in train_ds.targets]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(train_ds), replacement=True)
        shuffle = False
    pin = device.type == "cuda"
    keep = args.workers > 0  # epoch 마다 작업 프로세스를 새로 띄우지 않게 (Windows 에서 특히 느림)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=args.workers, pin_memory=pin, persistent_workers=keep)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin, persistent_workers=keep)

    model = build_model(len(GROUPS), args.arch, pretrained=not args.no_pretrained).to(device)
    if args.balance == "weight":
        criterion = nn.CrossEntropyLoss(weight=class_w.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    detail = "\n".join(f"{k}: {v}" for k, v in vars(args).items())
    config = f"설정: {aug_desc}\n\n{detail}\n클래스 가중치: {[round(float(w), 2) for w in class_w]}\n"
    (out_dir / "config.txt").write_text(config, encoding="utf-8")

    history = []
    best, best_epoch = -1.0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_bal = run_one_epoch(model, train_loader, criterion, optimizer, device, True, use_cutmix)
        va_loss, va_acc, va_bal = run_one_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()
        history.append([epoch, tr_loss, tr_acc, tr_bal, va_loss, va_acc, va_bal])
        print(f"[{epoch}/{args.epochs}] 학습 loss {tr_loss:.3f} acc {tr_acc:.1%} | "
              f"검증 loss {va_loss:.3f} acc {va_acc:.1%} bal.acc {va_bal:.1%} | "
              f"{time.time() - t0:.0f}초 (누적 {(time.time() - start) / 60:.1f}분)")

        # 모델 선택 기준은 정확도가 아니라 balanced accuracy
        if va_bal > best:
            best, best_epoch = va_bal, epoch
            torch.save({"model": model.state_dict(), "classes": GROUPS, "arch": args.arch,
                        "img_size": args.img_size, "aug": aug, "aug_desc": aug_desc, "balance": args.balance,
                        "epoch": epoch, "epochs": args.epochs}, out_dir / "best.pt")
            print(f"  -> best.pt 저장 (검증 balanced acc {best:.1%})")
    minutes = (time.time() - start) / 60

    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "train_bal_acc", "val_loss", "val_acc", "val_bal_acc"])
        w.writerows(history)
        w.writerow(["총 학습 시간(분)", f"{minutes:.1f}"])
    with open(out_dir / "config.txt", "a", encoding="utf-8") as f:
        f.write(f"best epoch: {best_epoch} (검증 balanced acc {best:.1%})\n총 학습 시간: {minutes:.1f}분\n")

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
    plt.close(fig)

    print(f"\n완료. 최고 검증 balanced accuracy {best:.1%} (epoch {best_epoch}), 총 {minutes:.1f}분. 결과: {out_dir}")
    print(f"다음: python src/evaluate.py --data {args.data} --ckpt {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
