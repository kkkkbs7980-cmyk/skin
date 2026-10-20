"""결과 폴더(기본 outputs_3cls_v3/) 아래 끝난 실험을 번호순으로 모아 결과표.md 하나로 만든다 (미팅용).

실행: python src/compare.py --out outputs_3cls_v3
만들어지는 것: <폴더>/결과표.md
    - 실험마다 "설정:" 줄(모델 / 적용 기법 / epoch·best epoch / 옵티마이저·배치·AMP) + 표. 괄호 = 원본 대비 %p
    - 맨 끝에 모델 비교표 (같은 50 epoch, ResNet50 vs ConvNeXt-Tiny, 원본·조합)
test_summary.json 이 없는 폴더(아직 안 돌았거나 도는 중, _smoke 연습 실행)는 건너뛴다.
"""
import argparse
import json
from pathlib import Path

from common import ARCH_KO, TABLE_NOTES, result_table

BASELINE_NAME = "00_baseline"
# 모델 비교표에 넣을 실험: (원본, 조합) 짝을 모델별로
MODEL_PAIRS = [("00_baseline", "05_combo"), ("06_resnet50_baseline", "07_resnet50_combo")]


def pct(v):
    return "–" if v is None else f"{v * 100:.1f}"


def model_compare_table(rows):
    by_name = {r["experiment"]: r for r in rows}
    lines = ["## 모델 비교 (같은 50 epoch·같은 학습 설정, ResNet50 vs ConvNeXt-Tiny)", "",
             "| 모델 | 증강 | balanced acc | accuracy | 악성 recall | 전체 F1 | 전체 AUC | best epoch |",
             "|---|---|---|---|---|---|---|---|"]
    n = 0
    for pair in MODEL_PAIRS:
        for name in pair:
            r = by_name.get(name)
            if r is None:
                continue
            n += 1
            aug = "조합" if "combo" in name else "원본"
            macro = r.get("macro", {})
            lines.append(f"| {ARCH_KO.get(r.get('arch'), r.get('arch', '?'))} | {aug} | "
                         f"{pct(r['balanced_accuracy'])} | {pct(r['accuracy'])} | "
                         f"{pct(r['recall']['malignant'])} | {pct(macro.get('f1'))} | {pct(macro.get('auc'))} | "
                         f"{r['best_epoch']}/{r['epochs']} |")
    if n == 0:
        return []
    lines += ["", "조합 = 뒤집기+회전+자르기+CutMix. 모델만 다르고 나머지(epoch·옵티마이저·배치·AMP)는 같음.", ""]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs_3cls_v3")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    out = Path(args.out)
    if not out.exists():
        print(f"{out} 폴더가 없음. 먼저 train.py / evaluate.py 실행")
        return
    rows = []
    for d in sorted(p for p in out.iterdir() if p.is_dir() and not p.name.startswith("_")):
        f = d / f"{args.split}_summary.json"
        if f.exists():
            rows.append(json.loads(f.read_text(encoding="utf-8")))
        else:
            print(f"건너뜀 (아직 결과 없음): {d.name}")
    if not rows:
        print("모을 결과가 없음. 먼저 evaluate.py 를 실행")
        return
    base = next((r for r in rows if r["experiment"] == BASELINE_NAME), None)

    md = ["# 3분류(정상/양성/악성) 실험 결과", "",
          f"{args.split} {rows[0]['n']}장 기준 (병변 단위 분할). 숫자는 %. "
          + ("괄호 = 원본(00_baseline) 대비 증가폭(%p)." if base else "원본 결과가 아직 없어 증가폭은 생략."),
          "", *TABLE_NOTES, ""]
    for r in rows:
        md += [f"## {r['experiment']}", "", result_table(r, base), ""]
    md += model_compare_table(rows)
    text = "\n".join(md)
    print(text)
    (out / "결과표.md").write_text(text, encoding="utf-8")
    print(f"\n저장: {out / '결과표.md'}  (실험 {len(rows)}개)")


if __name__ == "__main__":
    main()
