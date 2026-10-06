"""데이터가 제대로 받아졌는지 확인: split별·클래스별 이미지 개수를 센다.

실행: python src/check_data.py --data data
기대값: Training 800장, Validation 100장, Test 100장 × 15클래스
"""
import argparse
from pathlib import Path

from common import SPLIT_NAMES, find_split_dir, get_dataset, has_test_split

EXPECTED = {"train": 800, "val": 100, "test": 100}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="AIHUB 데이터 압축 푼 폴더")
    args = parser.parse_args()

    all_ok = True
    if not has_test_split(args.data):
        print("3.Test 폴더가 없음 (AIHUB는 test를 비공개). 아래처럼 나눠서 씀:")
        print("  제공된 Validation(100장/클래스) -> 우리 test")
        print("  제공된 Training(800장/클래스)  -> 우리 train 700장 + val 100장")
        EXPECTED["train"] = 700
    for split in SPLIT_NAMES:
        try:
            split_dir = find_split_dir(args.data, split)
        except FileNotFoundError as e:
            print(f"[{split}] {e}")
            all_ok = False
            continue

        print(f"\n[{split}] {split_dir}")
        ds = get_dataset(args.data, split, img_size=64)
        counts = {}
        for _, label in ds.samples:
            counts[ds.classes[label]] = counts.get(ds.classes[label], 0) + 1
        classes = [Path(split_dir) / c for c in ds.classes]
        total = 0
        for c in classes:
            n = counts.get(c.name, 0)
            total += n
            mark = "OK" if n == EXPECTED[split] else f"<-- 기대값 {EXPECTED[split]}"
            if n != EXPECTED[split]:
                all_ok = False
            print(f"  {c.name:12s} {n:5d}장  {mark}")
        print(f"  클래스 {len(classes)}개, 합계 {total}장")
        if len(classes) != 15:
            all_ok = False
            print("  <-- 클래스가 15개가 아님")

    print("\n결과:", "정상" if all_ok else "확인 필요 (위에 표시된 항목 보기)")


if __name__ == "__main__":
    main()
