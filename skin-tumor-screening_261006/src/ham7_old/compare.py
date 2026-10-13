"""실험 A/B/C 결과를 표 하나와 그림 하나로 묶는다 (미팅용).

실행: python src/compare.py            (outputs/ 아래 test_summary.json 이 있는 실험을 전부 모음)
      python src/compare.py --exps outputs/A_baseline outputs/B_aug outputs/C_aug_weight   (순서 지정)

만들어지는 것:
    outputs/비교표.md          정확도·balanced accuracy·macro F1·mel recall·악성 recall 표 (보고서에 붙여넣기)
    outputs/비교_recall.png    실험별 클래스 recall 막대그림
"""
import argparse
import json
from pathlib import Path

from common import CLASS_KO, CLASSES, setup_korean_font

# 실험별 고정 색 (순서 바뀌어도 같은 실험은 같은 색)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exps", nargs="*", help="실험 폴더들. 생략하면 outputs/ 아래 전부")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    if args.exps:
        dirs = [Path(d) for d in args.exps]
    else:
        dirs = sorted(p.parent for p in Path(args.out).glob("*/test_summary.json"))
    rows = []
    for d in dirs:
        f = d / "test_summary.json"
        if not f.exists():
            print(f"건너뜀 (test_summary.json 없음): {d}")
            continue
        rows.append(json.loads(f.read_text(encoding="utf-8")))
    if not rows:
        print("비교할 결과가 없음. 먼저 evaluate.py 를 실행")
        return

    def cond(r):
        parts = ["증강" if r["aug"] else "증강 없음"]
        if r["balance"] != "none":
            parts.append({"weight": "클래스 가중치", "sampler": "가중 샘플링"}[r["balance"]])
        return " + ".join(parts)

    head = ["실험", "조건", "정확도", "balanced acc", "macro F1", "mel recall", "악성·전암 recall", "양성 오경고"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        md.append("| " + " | ".join([
            r["experiment"], cond(r), f"{r['accuracy']:.1%}", f"{r['balanced_accuracy']:.1%}",
            f"{r['macro_f1']:.3f}", f"{r['recall']['mel']:.1%}", f"{r['malignant_recall']:.1%}",
            f"{r['benign_false_alarm']:.1%}"]) + " |")
    md.append("")
    md.append("클래스별 recall")
    md.append("| 실험 | " + " | ".join(CLASSES) + " |")
    md.append("|---|" + "---|" * len(CLASSES))
    for r in rows:
        md.append(f"| {r['experiment']} | " + " | ".join(f"{r['recall'][c]:.1%}" for c in CLASSES) + " |")
    md.append("")
    md.append(f"test {rows[0]['n']}장 (병변 단위 분할). 정확도는 전부 nv 로 찍어도 약 67% 가 나오므로 balanced accuracy 를 기준으로 비교.")
    text = "\n".join(md)
    print(text)
    out = Path(args.out)
    (out / "비교표.md").write_text(text, encoding="utf-8")

    setup_korean_font()
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(len(CLASSES))
    k = len(rows)
    width = 0.8 / k
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, r in enumerate(rows):
        vals = [r["recall"][c] for c in CLASSES]
        ax.bar(x + (i - (k - 1) / 2) * width, vals, width * 0.9, color=COLORS[i % len(COLORS)],
               label=f"{r['experiment']} ({cond(r)})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{CLASS_KO[c]}" for c in CLASSES], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("recall (실제 중 맞춘 비율)")
    ax.set_title("실험별 클래스 recall (test)")
    ax.yaxis.grid(True, color="#dddddd"); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "비교_recall.png", dpi=120)
    print(f"\n저장: {out / '비교표.md'}, {out / '비교_recall.png'}")


if __name__ == "__main__":
    main()
