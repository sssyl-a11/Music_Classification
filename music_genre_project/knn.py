import numpy as np


def predict_one(train_features, train_labels, sample, k):
    """计算距离，取最近的 k 个样本进行投票。"""
    if not 1 <= k <= len(train_labels):
        raise ValueError('k 必须在 1 和训练样本数之间')
    distances = np.sqrt(np.sum((train_features - sample) ** 2, axis=1))
    indices = np.argsort(distances, kind='stable')[:k]
    votes = {}
    for index in indices:
        label = train_labels[index]
        votes[label] = votes.get(label, 0) + 1
    # 票数相同时，选择最先出现的类别，即更近的邻居所属类别。
    return max(votes, key=votes.get)


def predict_batch(train_features, train_labels, test_features, k):
    predictions = []
    for sample in test_features:
        predictions.append(predict_one(train_features, train_labels, sample, k))
    return np.array(predictions)
