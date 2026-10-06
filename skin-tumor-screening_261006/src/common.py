"""여러 스크립트에서 같이 쓰는 함수 모음 (HAM10000 버전)."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# HAM10000 의 7가지 진단 코드와 우리말 이름. 이 순서가 곧 모델 출력 순서다.
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
# 악성·전암 그룹. 이 셋의 확률 합이 "악성 위험도"가 된다.
#   mel: 악성흑색종, bcc: 기저세포암 (악성)
#   akiec: 광선각화증·보웬병 (전암 / 상피내암 → 전문의 상담 필요)
MALIGNANT = ["mel", "bcc", "akiec"]

# ImageNet 으로 미리 학습된 모델이 기대하는 픽셀 정규화 값
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

SPLIT_FILE = "split.csv"  # prepare_split.py 가 data/ 안에 만드는 파일


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
    """data/ 아래 모든 jpg 를 찾아 {image_id: 경로} 로 돌려준다 (part_1, part_2 어디에 있든 상관없음)."""
    data_root = Path(data_root)
    found = {}
    for p in data_root.rglob("*.jpg"):
        found.setdefault(p.stem, p)
    return found


def load_split(data_root):
    """prepare_split.py 가 만든 split.csv 읽기."""
    path = Path(data_root) / SPLIT_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없음. 먼저 실행: python src/prepare_split.py --data {data_root}")
    df = pd.read_csv(path)
    return df


def get_transforms(img_size, aug):
    """aug=False: 크기만 맞춤(원본 실험). aug=True: 뒤집기·회전·색 변화·무작위 자르기(증강 실험)."""
    if aug:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class HamDataset(Dataset):
    """split.csv 의 한 split(train/val/test) 을 읽는 데이터셋."""

    def __init__(self, data_root, split, img_size, aug=False):
        df = load_split(data_root)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"split.csv 에 '{split}' 데이터가 없음")
        self.classes = CLASSES
        self.targets = [CLASSES.index(d) for d in self.df["dx"]]
        self.tf = get_transforms(img_size, aug)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        return self.tf(img), self.targets[i]

    def class_counts(self):
        return [self.targets.count(c) for c in range(len(CLASSES))]


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

    system = platform.system()
    if system == "Windows":
        matplotlib.rc("font", family="Malgun Gothic")
    elif system == "Darwin":
        matplotlib.rc("font", family="AppleGothic")
    else:
        matplotlib.rc("font", family="NanumGothic")
    matplotlib.rcParams["axes.unicode_minus"] = False
