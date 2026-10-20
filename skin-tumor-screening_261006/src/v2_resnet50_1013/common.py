"""여러 스크립트에서 같이 쓰는 함수 모음 (HAM10000 3분류 버전: 정상 / 양성 / 악성).

7종 분류 때 코드는 src/ham7_old/ 에 보관.
"""
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms

# HAM10000 원본의 7가지 진단 코드와 우리말 이름 (metadata 의 dx 열)
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_KO = {
    "akiec": "광선각화증/보웬병",
    "bcc": "기저세포암",
    "bkl": "양성각화증(지루각화증 등)",
    "df": "피부섬유종",
    "mel": "악성흑색종",
    "nv": "멜라닌세포모반",
    "vasc": "혈관성병변",
}

# 3분류 그룹. 이 순서가 곧 모델 출력 순서다.
GROUPS = ["normal", "benign", "malignant"]
GROUP_KO = {"normal": "정상(점)", "benign": "양성", "malignant": "악성"}
DX_TO_GROUP = {
    "nv": "normal",
    "bkl": "benign", "df": "benign", "vasc": "benign",
    "mel": "malignant", "bcc": "malignant", "akiec": "malignant",
}

# ImageNet 으로 미리 학습된 모델이 기대하는 픽셀 정규화 값
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

SPLIT_FILE = "split_3cls.csv"  # prepare_split.py 가 data/ 안에 만드는 파일 (7종용 split.csv 와 별개)
CACHE_DIR = "cache_320"        # data/cache_320/ : 짧은 변 320px 로 줄여 둔 사진


def find_metadata(data_root):
    """data/ 아래 어디에 있든 HAM10000_metadata(.csv/.tab) 파일을 찾는다."""
    data_root = Path(data_root)
    for pat in ("HAM10000_metadata*.csv", "HAM10000_metadata*", "*metadata*.csv"):
        hits = [p for p in data_root.rglob(pat) if p.is_file()]
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{data_root} 아래에서 HAM10000_metadata 파일을 못 찾음")


def read_metadata(path):
    """metadata 파일 읽기. 캐글은 쉼표, Harvard Dataverse는 탭으로 구분돼 있어 둘 다 처리."""
    df = pd.read_csv(path, sep=None, engine="python")
    need = {"lesion_id", "image_id", "dx"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"metadata 에 {missing} 열이 없음. 열 목록: {list(df.columns)}")
    return df


def find_images(data_root):
    """data/ 아래 원본 jpg 를 찾아 {image_id: 경로} 로 돌려준다 (축소 캐시 폴더는 제외)."""
    data_root = Path(data_root)
    found = {}
    for p in data_root.rglob("*.jpg"):
        if CACHE_DIR in p.parts:
            continue
        found.setdefault(p.stem, p)
    return found


def load_split(data_root):
    """prepare_split.py 가 만든 split_3cls.csv 읽기."""
    path = Path(data_root) / SPLIT_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없음. 먼저 실행: python src/prepare_split.py --data {data_root}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 증강 기법. 이름 하나 = 기법 하나. train.py --aug flip crop ... 로 골라 쓴다.
# ---------------------------------------------------------------------------
AUG_NAMES = ["flip", "crop", "color", "cutmix", "hair", "vignette", "erase", "elastic"]

AUG_DESC = {
    "flip": "방향: 좌우·상하 뒤집기 각 50%, 회전 0–360°",
    "crop": "자르기: 원본의 70–100% 영역을 잘라 224로",
    "color": "색 변화: 밝기·대비·채도 ±20%, 색조 ±2%",
    "cutmix": "CutMix: 배치 안 두 사진을 사각형으로 잘라 붙임, alpha=1.0",
    "hair": "가짜 털: 어두운 곡선 0–8개를 무작위로 그림(확률 50%)",
    "vignette": "렌즈 테두리: 가장자리를 둥글게 어둡게(확률 50%)",
    "erase": "가림: 2–15% 영역을 지움(확률 50%)",
    "elastic": "탄성 변형: alpha 30",
}
ELASTIC_FALLBACK_DESC = "탄성 변형 대신 기울이기(shear 10°) — torchvision 구버전이라 ElasticTransform 없음"
NO_AUG_DESC = "원본: 증강 없음 (크기만 224로 맞춤)"


class RandomHair:
    """사진 위에 어두운(검정–갈색) 곡선을 0–8개 그린다. 피부경 사진의 털을 흉내 냄."""

    def __init__(self, p=0.5, max_lines=8):
        self.p = p
        self.max_lines = max_lines

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for _ in range(random.randint(0, self.max_lines)):
            # 시작점에서 방향을 조금씩 틀며 여러 점을 이어 곡선을 만든다
            x, y = random.uniform(0, w), random.uniform(0, h)
            angle = random.uniform(0, 2 * math.pi)
            step = random.uniform(0.04, 0.10) * max(w, h)
            pts = [(x, y)]
            for _ in range(random.randint(3, 8)):
                angle += random.uniform(-0.5, 0.5)
                x += step * math.cos(angle)
                y += step * math.sin(angle)
                pts.append((x, y))
            dark = random.randint(0, 50)
            color = (dark + random.randint(0, 40), dark + random.randint(0, 20), dark)  # 검정~갈색
            draw.line(pts, fill=color, width=random.randint(1, 3), joint="curve")
        return img


class RandomVignette:
    """사진 중심에서 반지름 r(짧은 변의 0.45–0.6배) 밖을 부드럽게 검게. 피부경 렌즈 테두리 흉내."""

    def __init__(self, p=0.5, r_range=(0.45, 0.6), soft=0.08):
        self.p = p
        self.r_range = r_range
        self.soft = soft

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        a = np.asarray(img).astype(np.float32)
        h, w = a.shape[:2]
        short = min(h, w)
        r = random.uniform(*self.r_range) * short
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
        factor = np.clip(1 - (d - r) / (self.soft * short), 0, 1)
        return Image.fromarray((a * factor[..., None]).astype(np.uint8))


def make_elastic():
    """ElasticTransform 이 있으면 쓰고, 없으면(구버전) 작은 기울이기로 대체. (transform, 설명) 반환."""
    if hasattr(transforms, "ElasticTransform"):
        return transforms.ElasticTransform(alpha=30.0), AUG_DESC["elastic"]
    return transforms.RandomAffine(degrees=0, shear=10), ELASTIC_FALLBACK_DESC


def aug_description(aug):
    """config.txt / 결과표 위에 실을 사람이 읽는 한 줄."""
    if not aug:
        return NO_AUG_DESC
    parts = []
    for name in aug:
        parts.append(make_elastic()[1] if name == "elastic" else AUG_DESC[name])
    return " / ".join(parts)


def get_transforms(img_size, aug=()):
    """aug: 기법 이름 목록. 비어 있으면 원본(크기만 맞춤). cutmix 는 배치 단위라 train.py 에서 처리."""
    aug = set(aug or [])
    unknown = aug - set(AUG_NAMES)
    if unknown:
        raise ValueError(f"모르는 증강 이름: {unknown}. 가능: {AUG_NAMES}")
    tf = []
    if "hair" in aug:  # 원래 크기(320px) 사진에 털을 그린 뒤 줄인다
        tf.append(RandomHair(p=0.5))
    if "crop" in aug:
        tf.append(transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)))
    else:
        tf.append(transforms.Resize((img_size, img_size)))
    if "flip" in aug:
        tf += [transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5),
               transforms.RandomRotation(360)]
    if "elastic" in aug:
        tf.append(make_elastic()[0])
    if "color" in aug:
        tf.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.02))
    if "vignette" in aug:
        tf.append(RandomVignette(p=0.5))
    tf += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if "erase" in aug:
        tf.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)))
    return transforms.Compose(tf)


class HamDataset(Dataset):
    """split_3cls.csv 의 한 split(train/val/test) 을 읽는 데이터셋. 라벨 = 3그룹 번호."""

    def __init__(self, data_root, split, img_size, aug=()):
        df = load_split(data_root)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"{SPLIT_FILE} 에 '{split}' 데이터가 없음")
        self.classes = GROUPS
        self.targets = [GROUPS.index(g) for g in self.df["group"]]
        self.tf = get_transforms(img_size, aug)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        return self.tf(img), self.targets[i]

    def class_counts(self):
        return [self.targets.count(c) for c in range(len(GROUPS))]


def build_model(num_classes, arch="resnet50", pretrained=True):
    """미리 학습된 모델을 가져와 마지막 층만 우리 클래스 수에 맞게 바꾼다 (전이학습)."""
    from torchvision import models

    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"모르는 모델 이름: {arch}")
    return model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("경고: GPU를 못 찾아서 CPU로 돌립니다. 아주 느립니다.")
    return torch.device("cpu")


def setup_korean_font():
    """그래프에 한글이 깨지지 않게 폰트 설정 (Windows: 맑은 고딕, Mac: AppleGothic)."""
    import platform
    import matplotlib

    matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장
    system = platform.system()
    if system == "Windows":
        matplotlib.rc("font", family="Malgun Gothic")
    elif system == "Darwin":
        matplotlib.rc("font", family="AppleGothic")
    else:
        matplotlib.rc("font", family="NanumGothic")
    matplotlib.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 결과표 (evaluate.py 와 compare.py 가 같이 씀)
# ---------------------------------------------------------------------------
def _pct(v, base, show_delta):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    s = f"{v * 100:.1f}"
    if show_delta and base is not None and not (isinstance(base, float) and math.isnan(base)):
        s += f" ({(v - base) * 100:+.1f})"
    return s


def result_table(summary, base=None):
    """test_summary.json 하나 → 설정 한 줄 + 표(행=클래스, 열=지표). base 가 있으면 괄호에 원본 대비 %p."""
    show = base is not None and base.get("experiment") != summary.get("experiment")
    lines = [
        f"설정: {summary['aug_desc']}   ({summary['epochs']} epoch, best epoch {summary['best_epoch']})",
        "",
        "| | recall | precision | F1 | AUC | accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for g in GROUPS:
        cells = []
        for key in ("recall", "precision", "f1", "auc"):
            b = base[key][g] if show else None
            cells.append(_pct(summary[key][g], b, show))
        lines.append(f"| {GROUP_KO[g]} | " + " | ".join(cells) + " | – |")
    acc = _pct(summary["accuracy"], base["accuracy"] if show else None, show)
    lines.append(f"| 전체 | – | – | – | – | {acc} |")
    return "\n".join(lines)
