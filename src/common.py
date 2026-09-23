"""여러 스크립트에서 같이 쓰는 함수 모음."""
from pathlib import Path

import torch
from torchvision import datasets, transforms

# 악성 3종. 폴더명(질환명)이 곧 클래스 이름이다.
MALIGNANT = ["기저세포암", "편평세포암", "악성흑색종"]

# 데이터 폴더 안에서 split 폴더를 찾을 때 쓰는 이름
SPLIT_NAMES = {
    "train": ("Training", "1.Training"),
    "val": ("Validation", "2.Validation"),
    "test": ("Test", "3.Test"),
}

# ImageNet 으로 미리 학습된 모델이 기대하는 픽셀 정규화 값
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def find_split_dir(data_root, split):
    """data_root 아래에서 Training/01.원천데이터 같은 폴더를 찾아 준다.

    AIHUB 압축을 풀면 경로 깊이가 제각각이라 재귀로 찾는다.
    """
    data_root = Path(data_root)
    names = SPLIT_NAMES[split]
    split_dir = next((data_root / name for name in names if (data_root / name).is_dir()), None)
    if split_dir is None:
        split_dir = next((p for name in names for p in data_root.rglob(name) if p.is_dir()), None)
    if split_dir is None:
        raise FileNotFoundError(f"{names} 폴더를 {data_root} 아래에서 못 찾음")
    # 그 안에 '1.원천데이터'(png) 폴더가 있으면 그걸 쓴다. 라벨(json) 폴더는 쓰지 않는다.
    for sub in split_dir.iterdir():
        if sub.is_dir() and "원천" in sub.name:
            return sub
    return split_dir


def get_transforms(img_size, train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_dataset(data_root, split, img_size):
    """이미지 폴더명을 라벨로 쓰되 AI Hub 분할 접두사는 제거한다."""
    split_dir = find_split_dir(data_root, split)
    ds = datasets.ImageFolder(split_dir, transform=get_transforms(img_size, train=(split == "train")))
    prefixes = {"train": "TS_", "val": "VS_"}
    prefix = prefixes.get(split, "")
    if prefix and all(name.startswith(prefix) for name in ds.classes):
        ds.classes = [name[len(prefix):] for name in ds.classes]
        ds.class_to_idx = {name: i for i, name in enumerate(ds.classes)}
    return ds


def build_model(num_classes, arch="resnet50", pretrained=True):
    """미리 학습된 모델을 가져와 마지막 층만 우리 클래스 수에 맞게 바꾼다 (전이학습)."""
    from torchvision import models

    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet101":
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        model = models.resnet101(weights=weights)
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
    """그래프에 한글 클래스명이 깨지지 않게 폰트 설정 (Windows: 맑은 고딕, Mac: AppleGothic)."""
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
