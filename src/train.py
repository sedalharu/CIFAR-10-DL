# 3. Traning

# Architecture Condition:
# Pytorch의 torchvision.datasets를 이용하여 CIFAR-10 데이터셋 로드
# Random Crop, Flipping, Normalization과 같은 데이터 증강 기법

import torch
import torch.nn as nn
import torch.optim as optim
import time


def train_model(model, trainloader, testloader, num_epochs=150, learning_rate=0.001):
    # 가능하면 GPU를 사용하도록 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 다중 분류를 위한 크로스 엔트로피 손실
    criterion = nn.CrossEntropyLoss()

    # Adam 옵티마이저 사용
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    train_losses = []
    train_accs = []
    test_accs = []
    best_acc = 0.0  # 최고 정확도

    # epoch 단위 학습
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # batch 단위 학습
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # 순전파 (forward pass)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 역전파 (backward pass)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            # 가장 높은 확률을 가진 클래스 선택
            _, predicted = outputs.max(1)

            # 전체 샘플 수 누적
            total += labels.size(0)

            # 정확한 예측 수 누적
            correct += predicted.eq(labels).sum().item()

        # epoch training performance 계산
        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()  # 평가 모드로 설정
        correct = 0
        total = 0

        # 테스트 시에는 역전파가 필요 없으므로 Gradient 비활성화
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # 검증 정확도 계산 및 기록
        test_acc = 100. * correct / total
        test_accs.append(test_acc)

        # 현재 epoch의 결과 출력
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Acc: {test_acc:.2f}% | Time: {epoch_time:.1f}s')

        # 최고 성능 모델 저장
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with accuracy: {best_acc:.2f}%')

        print('-' * 70)

    return train_losses, train_accs, test_accs


def plot_training_results(train_losses, train_accs, test_accs):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))

    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results.png')  # 이미지 저장 추가
    plt.close()  # 메모리 관리를 위해 figure 닫기