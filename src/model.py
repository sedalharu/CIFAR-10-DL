# 2. Model Implementation

# Architecture Condition:
# 최소 3개 CNN 아키텍처 (conv, batchnorm, relu, maxpool), Fully connected layers
# 과적합 방지를 위한 Dropout 포함

# Input: 3x32x32 (CIFAR-10 이미지)
# Output: 10 (클래스 수)

import torch
import torch.nn as nn


class CIFAR10CNN(nn.Module):

    def __init__(self):
        super(CIFAR10CNN, self).__init__()

        # Convolutional Layer 첫 번째
        self.conv1 = nn.Sequential(
            # 3채널 입력 -> 32채널로 확장, 3x3 커널 사용
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # 배치 정규화: 학습 안정화 및 과적합 방지 용도
            nn.BatchNorm2d(32),
            # ReLU 추가
            nn.ReLU(),
            # 특징 추출을 위한 맵 크기를 절반으로 축소 (32x32 -> 16x16)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 입력: 3 x 32 x 32 -> 출력: 32 x 16 x 16

        # Convolutional Layer 두 번째
        self.conv2 = nn.Sequential(
            # 32채널 입력 -> 64채널로 확장
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 특징맵 크기를 다시 절반으로 축소 (16x16 -> 8x8)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 입력: 32 x 16 x 16 -> 출력: 64 x 8 x 8

        # Convolutional Layer 세 번째
        self.conv3 = nn.Sequential(
            # 채널 수를 64에서 128로 증가
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 특징맵 크기를 다시 1/2로 축소 (8x8 -> 4x4)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 입력: 64 x 8 x 8 -> 출력: 128 x 4 x 4

        # Fully Connected Layer
        self.fc = nn.Sequential(
            # Dropout 첫 번째: 과적합 방지
            nn.Dropout(0.5),
            # 128*4*4=2048 크기의 특징맵을 512 유닛으로 연결
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            # Dropout 두 번째: 추가적인 정규화
            nn.Dropout(0.5),
            # 최종 출력층: Dataset 10개 클래스에 대한 로짓 출력
            nn.Linear(512, 10)
        )

    # Forward Propagation 함수
    def forward(self, x):
        # Convolutional Layer 통과
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # 특징맵 flatten
        # (batch_size x 128 x 4 x 4) -> (batch_size x 2048)
        x = x.view(x.size(0), -1)

        # Fully Connected 레이어 통과
        x = self.fc(x)
        return x

    # Model Weight 초기화
    def initialize_weights(self):
        for m in self.modules():
            # Conv2d
            if isinstance(m, nn.Conv2d):
                # He 초기화: 층이 깊어질수록 기울기가 사라지거나 폭발하는 문제 방지
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 편향이 있는 경우
                if m.bias is not None:
                    # 편향을 0으로 초기화
                    nn.init.constant_(m.bias, 0)

            # BatchNorm2d
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            # FullyConnected
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)