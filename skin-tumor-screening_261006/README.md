# HAM10000 피부종양 분류 실험 (2026-10-06)

이번 HAM10000 실험의 코드와 결과 공유용 폴더입니다. 저장소 루트의 기존 AIHUB/PAD-UFES-20 실험과 구분하여 보관합니다.

- `src/`: HAM10000 데이터 확인, 병변 단위 분할, 학습, 평가, 비교 코드. `src/aihub_old/`는 이전 코드 보관용입니다.
- `outputs/`: A/B/C 모델 가중치(`best.pt`), 학습 이력, 시험 평가 결과, 혼동행렬, 비교표와 그래프.
- 원본 데이터는 포함하지 않습니다. HAM10000을 이 폴더의 `data/` 아래에 압축 해제하면 됩니다.

## 시험 결과

동일한 병변 단위 분할의 시험 이미지 1,987장으로 평가했습니다. 공통 설정은 ImageNet 사전학습 ResNet50, 입력 224px, 배치 32, 10 epoch, Adam 초기 학습률 1e-4입니다. 각 실험의 최종 평가에는 검증 balanced accuracy가 가장 높았던 모델을 사용했습니다.

| 실험 | 조건 | 정확도 | Balanced accuracy | Macro F1 | mel recall |
|---|---|---:|---:|---:|---:|
| A | 증강 없음 | 83.39% | 69.67% | 0.696 | 48.09% |
| B | 증강 | 84.70% | 75.42% | 0.749 | 54.47% |
| C | 증강 + 클래스 가중치 | 79.62% | 75.92% | 0.662 | 59.15% |

증강으로 전반적인 성능이 개선되었습니다. 클래스 가중치를 추가하면 mel 및 악성·전암 그룹 recall이 상승했으나 양성 오경고율도 증가했습니다. 단일 분할·seed의 결과입니다.

[전체 비교표](outputs/비교표.md)

![클래스별 recall 비교](outputs/비교_recall.png)

## 실행

이 폴더를 작업 디렉터리로 사용합니다. GPU용 PyTorch와 torchvision이 설치된 환경에서 나머지 의존성을 설치합니다.

```bat
python -m pip install -r requirements.txt
python src/check_data.py --data data
python src/prepare_split.py --data data
python src/evaluate.py --data data --ckpt outputs/A_baseline/best.pt
python src/evaluate.py --data data --ckpt outputs/B_aug/best.pt
python src/evaluate.py --data data --ckpt outputs/C_aug_weight/best.pt
python src/compare.py
```

원래 실험의 `split.csv`는 포함하지 않습니다. 분할을 재현하려면 동일한 HAM10000 메타데이터와 이미지, 기본 seed 0을 사용하고 클래스별 이미지 수 및 병변 수를 먼저 확인합니다.
