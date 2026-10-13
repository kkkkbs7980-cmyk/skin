"""HAM10000 을 3분류(정상/양성/악성)용으로 고르고 나눠 data/split_3cls.csv 를 만든다. (한 번만 실행)

실행: python src/prepare_split.py --data data

3그룹:
  normal 정상(점)  = nv              → 6,705장 중 약 2,000장만 (병변 단위로 무작위 추출)
  benign 양성      = bkl + df + vasc → 전부 (1,356장)
  malignant 악성   = mel + bcc + akiec → 전부 (1,954장)

왜 병변 단위로 나누나:
  같은 병변(lesion_id)을 여러 번 찍은 사진이 많다. 사진 단위로 나누면 같은 병변이 학습과 시험에
  동시에 들어가서 점수가 부풀려진다. 그래서 정상 추출도, train/val/test 분할도 전부 병변 단위로 한다.
  비율: 병변 기준 train 60% / val 20% / test 20%, 진단(dx)별로 따로 나눔(층화), seed 0.

속도용 사전 축소: 쓰는 사진을 짧은 변 320px(600×450 → 427×320), JPG 품질 95 로
  data/cache_320/ 에 한 번 저장해 둔다 (원본은 그대로). 모델 입력이 224 라 성능 손실 없음.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import (CACHE_DIR, CLASSES, DX_TO_GROUP, GROUP_KO, GROUPS, SPLIT_FILE, find_images, find_metadata,
                    read_metadata)


def sample_normal(df, n_target, rng):
    """nv 병변을 무작위 순서로 하나씩 추가해 사진 장수가 n_target 에 닿으면 멈춘다 (사진 단위 추출 금지)."""
    nv = df[df["dx"] == "nv"]
    lesion_ids = nv["lesion_id"].drop_duplicates().tolist()
    rng.shuffle(lesion_ids)
    per_lesion = nv.groupby("lesion_id").size()
    chosen, total = [], 0
    for lid in lesion_ids:
        if total >= n_target:
            break
        chosen.append(lid)
        total += int(per_lesion[lid])
    return set(chosen)


def make_cache(df, data_root, short_side):
    """사진을 짧은 변 short_side 로 줄여 data/cache_320/<image_id>.jpg 에 저장. 이미 있으면 건너뜀."""
    cache = Path(data_root) / CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    new_paths, made = [], 0
    for image_id, src in tqdm(zip(df["image_id"], df["path"]), total=len(df), desc="사진 축소 캐시"):
        dst = cache / f"{image_id}.jpg"
        if not dst.exists():
            img = Image.open(src).convert("RGB")
            w, h = img.size
            s = short_side / min(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
            img.save(dst, quality=95)
            made += 1
        new_paths.append(str(dst))
    print(f"캐시: 새로 {made}장 저장, {len(df) - made}장은 이미 있어서 건너뜀 → {cache}")
    return new_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="HAM10000 압축 푼 폴더")
    parser.add_argument("--normal-n", type=int, default=2000, help="정상(nv) 사진을 몇 장 쓸지 (병변 단위라 대략)")
    parser.add_argument("--val", type=float, default=0.20)
    parser.add_argument("--test", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=320, help="사전 축소 짧은 변 길이(px)")
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

    # 1) 정상(nv) 일부만 병변 단위로 고르기
    rng = np.random.RandomState(args.seed)
    keep_nv = sample_normal(df, args.normal_n, rng)
    df = df[(df["dx"] != "nv") | df["lesion_id"].isin(keep_nv)].reset_index(drop=True)
    df["group"] = df["dx"].map(DX_TO_GROUP)
    print(f"정상(nv): 병변 {len(keep_nv)}개, 사진 {(df['dx'] == 'nv').sum()}장 선택 (목표 {args.normal_n}장)")

    # 2) 병변 단위 60/20/20 분할, dx 별 층화
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

    # 같은 병변이 두 split 에 걸쳐 있으면 실패
    leak = df.groupby("lesion_id")["split"].nunique().max()
    if leak != 1:
        raise RuntimeError("같은 병변이 여러 split 에 들어감 (버그)")

    # 3) 사전 축소 캐시. split 파일의 path 는 캐시를 가리킨다.
    df["path"] = make_cache(df, args.data, args.cache_size)

    out = Path(args.data) / SPLIT_FILE
    df[["image_id", "lesion_id", "dx", "group", "split", "path"]].to_csv(out, index=False, encoding="utf-8")

    # 확인용 표
    print(f"\n병변 {len(lesions)}개, 사진 {len(df)}장을 아래처럼 나눔 (사진 장수 기준):")
    table = df.pivot_table(index="group", columns="split", values="image_id", aggfunc="count", fill_value=0)
    table = table.reindex(GROUPS)[["train", "val", "test"]]
    table["합계"] = table.sum(1)
    table.index = [f"{g} {GROUP_KO[g]}" for g in table.index]
    print(table.to_string())
    print(f"합계: train {int(table['train'].sum())} / val {int(table['val'].sum())} / test {int(table['test'].sum())}")
    print(f"\n저장: {out}   (병변 겹침 없음 확인)")
    print("다음: python src/train.py --data data --out outputs_3cls/00_baseline")


if __name__ == "__main__":
    main()
