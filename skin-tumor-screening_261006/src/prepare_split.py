"""HAM10000 을 학습/검증/시험으로 나눠 data/split.csv 를 만든다. (딱 한 번만 실행)

실행: python src/prepare_split.py --data data

왜 따로 나누나:
  HAM10000 은 같은 병변(lesion_id)을 여러 번 찍은 사진이 많다 (10,015장이지만 병변은 7,470개).
  사진 단위로 무작위로 나누면 같은 병변이 학습과 시험에 동시에 들어가서 정확도가 부풀려진다.
  그래서 '병변' 단위로 나누고, 클래스 비율이 비슷하게 유지되도록(층화) 클래스별로 따로 나눈다.
  val 은 "몇 번째 epoch 모델을 고를지" 정하는 데만 쓰고, test 는 마지막 평가에만 쓴다 (학습·선택에 절대 안 씀).

비율: 병변 기준 train 60% / val 20% / test 20%  (조교 기준·일반 관행)  (seed 고정 → 누가 돌려도 같은 결과)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import CLASSES, CLASS_KO, SPLIT_FILE, find_images, find_metadata, read_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="HAM10000 압축 푼 폴더")
    parser.add_argument("--val", type=float, default=0.20)
    parser.add_argument("--test", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    meta_path = find_metadata(args.data)
    df = read_metadata(meta_path)
    print(f"metadata: {meta_path}  ({len(df)}행)")

    images = find_images(args.data)
    df["path"] = df["image_id"].map(lambda i: str(images.get(i, "")))
    missing = (df["path"] == "").sum()
    if missing:
        print(f"경고: 이미지 파일을 못 찾은 행 {missing}개 → 제외함. (part_1, part_2 압축을 다 풀었는지 확인)")
        df = df[df["path"] != ""].reset_index(drop=True)

    unknown = set(df["dx"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"모르는 dx 값: {unknown}")

    # 병변 하나 = 진단 하나. 병변 목록을 클래스별로 섞어서 나눈다.
    rng = np.random.RandomState(args.seed)
    lesions = df.drop_duplicates("lesion_id")[["lesion_id", "dx"]]
    assign = {}
    for dx in CLASSES:
        ids = lesions.loc[lesions["dx"] == dx, "lesion_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, int(round(n * args.test)))
        n_val = max(1, int(round(n * args.val)))
        for k, lid in enumerate(ids):
            if k < n_test:
                assign[lid] = "test"
            elif k < n_test + n_val:
                assign[lid] = "val"
            else:
                assign[lid] = "train"
    df["split"] = df["lesion_id"].map(assign)

    out = Path(args.data) / SPLIT_FILE
    df[["image_id", "lesion_id", "dx", "split", "path"]].to_csv(out, index=False, encoding="utf-8")

    # 확인용 표
    print(f"\n병변 {len(lesions)}개, 사진 {len(df)}장을 아래처럼 나눔 (사진 장수 기준):")
    table = df.pivot_table(index="dx", columns="split", values="image_id", aggfunc="count", fill_value=0)
    table = table.reindex(CLASSES)[["train", "val", "test"]]
    table["합계"] = table.sum(1)
    table.index = [f"{c} {CLASS_KO[c]}" for c in table.index]
    print(table.to_string())
    print(f"합계: train {int(table['train'].sum())} / val {int(table['val'].sum())} / test {int(table['test'].sum())}")

    # 같은 병변이 두 split 에 걸쳐 있지 않은지 검사
    leak = df.groupby("lesion_id")["split"].nunique().max()
    assert leak == 1, "같은 병변이 여러 split 에 들어감 (버그)"
    print(f"\n저장: {out}   (병변 겹침 없음 확인)")
    print("다음: python src/train.py --data data --out outputs/A_baseline")


if __name__ == "__main__":
    main()
