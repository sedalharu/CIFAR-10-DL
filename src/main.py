import torch
from model import CIFAR10CNN
from data import trainloader, testloader
from train import train_model, plot_training_results
from evaluate import evaluate_model, plot_confusion_matrix, save_evaluation_results, save_model


def main():
    # GPU-CUDA 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 초기화
    model = CIFAR10CNN()
    model.initialize_weights()
    print("Model initialized with He initialization")

    # Training hyperparameter 설정
    num_epochs = 150
    learning_rate = 0.001

    print("\nStarting training with the following parameters:")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {trainloader.batch_size}")
    print("-" * 70)

    # 모델 훈련
    train_losses, train_accs, test_accs = train_model(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )

    # 훈련 결과 시각화
    plot_training_results(train_losses, train_accs, test_accs)

    print("\nTraining completed!")
    print(f"Best test accuracy: {max(test_accs):.2f}%")

    # 모델 평가
    print("\nEvaluating model...")
    metrics = evaluate_model(model, testloader, device)

    # 결과 출력 및 저장
    print(f"\nFinal Model Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")

    # Confusion Matrix 시각화
    plot_confusion_matrix(metrics['confusion_matrix'])

    # 평가 결과 저장
    save_evaluation_results(metrics)

    # 모델 저장
    save_model(model)


if __name__ == "__main__":
    main()