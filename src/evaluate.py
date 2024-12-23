# 4. Evaluation

# Architecture Condition:
# 테스트 데이터셋에서 모델을 평가하고 정확도, 정밀도, 재현율, F1-Score를 계산
# 훈련된 모델 저장

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from data import classes


# 성능 지표 시각화
def plot_class_performance(metrics):
    precisions = metrics['class_precision']
    recalls = metrics['class_recall']
    f1_scores = metrics['class_f1']

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(15, 6))
    plt.bar(x - width, precisions, width, label='Precision')
    plt.bar(x, recalls, width, label='Recall')
    plt.bar(x + width, f1_scores, width, label='F1-Score')

    plt.ylabel('Scores')
    plt.title('Performance Metrics by Class')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 그래프를 이미지 파일로 저장
    plt.savefig('class_performance.png')
    plt.close()


# 모델 평가
def evaluate_model(model, testloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = outputs.max(1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # numpy 배열 변환
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 성능 지표 계산
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds,
                                                               average='weighted')

    # 클래스별 성능 지표
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(all_labels,
                                                                                 all_preds,
                                                                                 average=None)

    # Confusion Matrix 계산
    confusion_mat = torch.zeros(len(classes), len(classes))
    for t, p in zip(all_labels, all_preds):
        confusion_mat[t, p] += 1

    # 추가된 시각화 함수 호출
    plot_class_performance({
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1
    })

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'confusion_matrix': confusion_mat
    }


# Confusion Matrix 시각화
def plot_confusion_matrix(confusion_mat):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt='.0f',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


# 평과 결과 저장
def save_evaluation_results(metrics, save_path='evaluation_results.txt'):
    with open(save_path, 'w') as f:
        f.write("모델 평가 결과\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Overall Precision: {metrics['precision']:.4f}\n")
        f.write(f"Overall Recall: {metrics['recall']:.4f}\n")
        f.write(f"Overall F1-Score: {metrics['f1']:.4f}\n\n")

        f.write("클래스별 성능:\n")
        f.write("-" * 50 + "\n")
        for i, class_name in enumerate(classes):
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {metrics['class_precision'][i]:.4f}\n")
            f.write(f"  Recall: {metrics['class_recall'][i]:.4f}\n")
            f.write(f"  F1-Score: {metrics['class_f1'][i]:.4f}\n")


# 학습된 모델 저장
def save_model(model, save_path='final_model.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model),
    }, save_path)
    print(f"Model saved to {save_path}")