import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from config import FEATURE_FILE, MODEL_FILE, OUTPUT_DIR, GENRES, RANDOM_SEED, K
from knn import predict_batch


def main():
    parser = argparse.ArgumentParser(description='训练和评估基础 KNN 分类器')
    parser.add_argument('--k', type=int, default=K)
    args = parser.parse_args()
    if not FEATURE_FILE.exists():
        raise FileNotFoundError('请先运行 python prepare_data.py')
    with np.load(FEATURE_FILE) as data:
        features = data['features']
        labels = data['labels']
        filenames = data['filenames']
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=RANDOM_SEED)
    train_features = features[train_indices]
    test_features = features[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    # 只用训练集计算均值和标准差，避免测试集信息泄漏。
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0] = 1
    train_features = (train_features - mean) / std
    test_features = (test_features - mean) / std
    predictions = predict_batch(train_features, train_labels, test_features, args.k)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, labels=GENRES,
                                   digits=4, zero_division=0)
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.savez(MODEL_FILE, train_features=train_features, train_labels=train_labels,
             mean=mean, std=std, k=args.k)
    (OUTPUT_DIR / 'classification_report.txt').write_text(report, encoding='utf-8')
    metrics = {'accuracy': accuracy, 'train_size': len(train_indices),
               'test_size': len(test_indices), 'k': args.k, 'random_seed': RANDOM_SEED}
    (OUTPUT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    with (OUTPUT_DIR / 'predictions.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'actual', 'predicted', 'correct'])
        for index, actual, predicted in zip(test_indices, test_labels, predictions):
            writer.writerow([filenames[index], actual, predicted, actual == predicted])
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay.from_predictions(test_labels, predictions, labels=GENRES,
        xticks_rotation=45, cmap='Blues', ax=ax, colorbar=False)
    ax.set_title('Music Genre Classification - KNN')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150)
    plt.close(fig)
    print(f'训练样本：{len(train_indices)}，测试样本：{len(test_indices)}')
    print(f'测试准确率：{accuracy:.2%}\n{report}')


if __name__ == '__main__':
    main()
