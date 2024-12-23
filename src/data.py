# 1. Data Preprocessing

# Architecture Condition:
# Pytorch의 torchvision.datasets를 이용하여 CIFAR-10 데이터셋 로드
# Random Crop, Flipping, Normalization과 같은 데이터 증강 기법

import os
import torch
import torchvision
import torchvision.transforms as transforms

# 데이터 저장 경로 설정
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_PATH, exist_ok=True)

# 학습 데이터에 대한 변환(augmentation) 정의
train_transform = transforms.Compose([
    # Random Crop: 32x32 이미지에 4픽셀 패딩을 추가, 그 후 랜덤하게 32x32 크기로 크롭
    transforms.RandomCrop(32, padding=4),

    # Horizontal Flipping: 이미지 좌우 반전
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    # 각 채널별, 평균과 표준편차를 사용하여 정규화
    transforms.Normalize(
        mean=(0.49139968, 0.48215827, 0.44653124),
        std=(0.24703233, 0.24348505, 0.26158768)
    )
])

# 테스트 시에는 데이터 증강을 하지 않고 정규화만 수행
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

batch_size = 64

# 멀티 스레딩을 위한 num_workers
num_workers = 2

# 학습 데이터셋 로드
trainset = torchvision.datasets.CIFAR10(
    root=DATA_PATH,
    train=True,
    download=True,
    transform=train_transform
)

# 학습 데이터 로더 생성
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,  # 배치 크기 설정
    shuffle=True,  # epoch 마다 데이터 순서를 shuffle
    num_workers=num_workers,
    pin_memory=True
)

# 테스트 데이터셋 로드
testset = torchvision.datasets.CIFAR10(
    root=DATA_PATH,
    train=False,
    download=True,
    transform=test_transform
)

# 테스트 데이터 로더 생성
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,  # 테스트 시에는 섞지 않음
    num_workers=num_workers,
    pin_memory=True
)

# CIFAR-10 데이터셋의 클래스 레이블
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')