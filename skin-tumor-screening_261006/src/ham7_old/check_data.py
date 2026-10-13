"""HAM10000 이 제대로 받아졌는지 확인: metadata 와 이미지 파일 개수를 센다.

실행: python src/check_data.py --data data
기대값: metadata 10,015행, 이미지 10,015장, 클래스 7개
"""
import argparse

from common import CLASSES, CLASS_KO, find_images, find_metadata, read_metadata

EXPECTED = {"akiec": 327, "bcc": 514, "bkl": 1099, "df": 115, "mel": 1113, "nv": 6705, "vasc": 142}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="HAM10000 압축 푼 폴더")
    args = parser.parse_args()

    ok = True
    try:
        meta_path = find_metadata(args.data)
    except FileNotFoundError as e:
        print(e)
        print("캐글(kmader/skin-cancer-mnist-ham10000) 압축을 data/ 안에 풀었는지 확인")
        return
    df = read_metadata(meta_path)
    print(f"metadata: {meta_path}")
    print(f"  {len(df)}행, 병변 {df['lesion_id'].nunique()}개  (기대: 10015행, 7470개)")

    images = find_images(args.data)
    print(f"이미지 파일: {len(images)}장  (기대: 10015장)")
    not_found = [i for i in df["image_id"] if i not in images]
    if not_found:
        ok = False
        print(f"  <-- metadata 에는 있는데 파일이 없는 사진 {len(not_found)}장. 예: {not_found[:3]}")
        print("      HAM10000_images_part_1 과 part_2 둘 다 풀었는지 확인")

    print("\n클래스별 사진 장수:")
    counts = df["dx"].value_counts()
    for c in CLASSES:
        n = int(counts.get(c, 0))
        mark = "OK" if n == EXPECTED[c] else f"<-- 기대값 {EXPECTED[c]}"
        if n != EXPECTED[c]:
            ok = False
        print(f"  {c:6s} {CLASS_KO[c]:16s} {n:5d}장  {mark}")
    extra = set(counts.index) - set(CLASSES)
    if extra:
        ok = False
        print(f"  <-- 모르는 클래스: {extra}")

    print("\n참고: nv(모반)가 전체의 67%. 전부 nv 라고 찍어도 정확도 67%가 나오므로")
    print("      정확도만 보지 말고 balanced accuracy 와 클래스별 recall 을 같이 봐야 한다.")
    print("\n결과:", "정상" if ok else "확인 필요 (위에 표시된 항목 보기)")
    if ok:
        print("다음: python src/prepare_split.py --data data")


if __name__ == "__main__":
    main()
