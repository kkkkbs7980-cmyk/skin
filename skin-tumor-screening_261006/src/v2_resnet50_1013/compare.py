"""outputs_3cls/ 아래 끝난 실험을 번호순으로 모아 결과표.md 하나로 만든다 (미팅용).

실행: python src/compare.py
만들어지는 것: outputs_3cls/결과표.md   (실험마다 설정 한 줄 + 표. 괄호 = 원본 대비 %p)
test_summary.json 이 없는 폴더(아직 안 돌았거나 도는 중)는 건너뛴다.
"""
import argparse
import json
from pathlib import Path

from common import result_table

BASELINE_NAME = "00_baseline"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs_3cls")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    out = Path(args.out)
    if not out.exists():
        print(f"{out} 폴더가 없음. 먼저 train.py / evaluate.py 실행")
        return
    rows = []
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
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
          + ("괄호 = 원본(00_baseline) 대비 증가폭(%p)." if base else "원본 결과가 아직 없어 증가폭은 생략."), ""]
    for r in rows:
        md += [f"## {r['experiment']}", "", result_table(r, base), ""]
    text = "\n".join(md)
    print(text)
    (out / "결과표.md").write_text(text, encoding="utf-8")
    print(f"\n저장: {out / '결과표.md'}  (실험 {len(rows)}개)")


if __name__ == "__main__":
    main()
