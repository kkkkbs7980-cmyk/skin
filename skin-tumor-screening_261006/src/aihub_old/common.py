"""여러 스크립트에서 같이 쓰는 함수 모음."""
from pathlib import Path

import torch
from torchvision import datasets, transforms

# 악성 3종. 폴더명(질환명)이 곧 클래스 이름이다.
MALIGNANT = ["기저세포암", "편평세포암", "악성흑색종"]

# 데이터 폴더 안에서 split 폴더를 찾을 때 쓰는 이름
SPLIT_NAMES = {
    "train": "1.Training",
    "val": "2.Validation",
    "test": "3.Test",
}

# ImageNet 으로 미리 학습된 모델이 기대하는 픽셀 정규화 값
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _find_dir(data_root, name):
    """data_root 아래 어디에 있든 name 폴더를 찾고, 그 안의 '1.원천데이터'(png) 폴더를 돌려준다. 없으면 None."""
    data_root = Path(data_root)
    candidates = [p for p in data_root.rglob(name) if p.is_dir()]
    if not candidates:
        return None
    split_dir = candidates[0]
    for sub in split_dir.iterdir():
        if sub.is_dir() and "원천" in sub.name:
            return sub
    return split_dir


def has_test_split(data_root):
    return _find_dir(data_root, SPLIT_NAMES["test"]) is not None


def find_split_dir(data_root, split):
    """split('train'/'val'/'test')에 해당하는 이미지 폴더를 찾는다.

    AIHUB에서 실제로 받은 데이터에는 3.Test 폴더가 없다 (문서에는 8:1:1이지만 test는 비공개).
    그래서 test 폴더가 없으면 아래처럼 나눠 쓴다.
        제공된 2.Validation(클래스당 100장) -> 우리 test (학습·모델 선택에 절대 안 씀)
        제공된 1.Training(클래스당 800장)  -> 우리 train 700장 + val 100장 (파일명 정렬 후 8장마다 1장을 val)
    """
    if has_test_split(data_root):
        d = _find_dir(data_root, SPLIT_NAMES[split])
    elif split == "test":
        d = _find_dir(data_root, SPLIT_NAMES["val"])
    else:
        d = _find_dir(data_root, SPLIT_NAMES["train"])
    if d is None:
        raise FileNotFoundError(f"'{SPLIT_NAMES[split]}' 폴더를 {data_root} 아래에서 못 찾음")
    return d


def _is_holdout_val(path):
    """Training 폴더를 train/val로 나눌 때 val로 보낼 파일인지. 같은 클래스 안에서 정렬 순서 8번째마다 1장."""
    return path.name in _holdout_names(path.parent)


_holdout_cache = {}


def _holdout_names(class_dir):
    class_dir = Path(class_dir)
    if class_dir not in _holdout_cache:
        files = sorted(p.name for p in class_dir.glob("*.png"))
        _holdout_cache[class_dir] = set(files[7::8])  # 8, 16, 24, ... 번째 파일
    return _holdout_cache[class_dir]


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
    """폴더명을 라벨로 쓰는 표준 방식(ImageFolder). test 폴더가 없으면 find_split_dir 설명대로 나눈다."""
    split_dir = find_split_dir(data_root, split)
    tf = get_transforms(img_size, train=(split == "train"))
    if has_test_split(data_root) or split == "test":
        return datasets.ImageFolder(split_dir, transform=tf)
    # Training 폴더를 train/val로 나누는 경우
    if split == "val":
        keep = lambda path: _is_holdout_val(Path(path))
    else:
        keep = lambda path: not _is_holdout_val(Path(path))
    return datasets.ImageFolder(split_dir, transform=tf,
                                is_valid_file=lambda path: path.lower().endswith(".png") and keep(path))


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
