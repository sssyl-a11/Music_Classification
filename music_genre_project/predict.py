import argparse
from pathlib import Path

import numpy as np

from config import MODEL_FILE
from features import extract_features
from knn import predict_one


def main():
    parser = argparse.ArgumentParser(description='预测一首 WAV 音乐的流派')
    parser.add_argument('audio', type=Path, help='WAV 文件路径')
    args = parser.parse_args()
    if not MODEL_FILE.exists():
        raise FileNotFoundError('请先运行 python train.py')
    sample = extract_features(args.audio)
    with np.load(MODEL_FILE) as model:
        sample = (sample - model['mean']) / model['std']
        prediction = predict_one(model['train_features'], model['train_labels'],
                                 sample, int(model['k']))
    print(f'文件：{args.audio.name}\n预测流派：{prediction}')


if __name__ == '__main__':
    main()
