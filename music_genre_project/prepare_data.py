import argparse
from pathlib import Path

import numpy as np

from config import DATA_DIR, OUTPUT_DIR, FEATURE_FILE, GENRES
from features import extract_features


def main():
    parser = argparse.ArgumentParser(description='从 WAV 文件提取 MFCC 特征')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    args = parser.parse_args()
    features, labels, filenames, errors = [], [], [], []
    for genre in GENRES:
        audio_files = sorted((args.data_dir / genre).glob('*.wav'))
        if not audio_files:
            raise ValueError(f'没有找到音频：{args.data_dir / genre}')
        count = 0
        for audio_path in audio_files:
            try:
                values = extract_features(audio_path)
            except (ValueError, OSError, EOFError) as error:
                errors.append(f'{audio_path.name}: {error}')
                continue
            features.append(values)
            labels.append(genre)
            filenames.append(f'{genre}/{audio_path.name}')
            count += 1
        print(f'{genre}: 成功处理 {count}/{len(audio_files)} 首', flush=True)
        if count < 3:
            raise ValueError(f'{genre} 有效音频太少，无法划分数据')
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.savez(FEATURE_FILE, features=features, labels=labels, filenames=filenames)
    (OUTPUT_DIR / 'skipped_files.txt').write_text('\n'.join(errors), encoding='utf-8')
    print(f'共保存 {len(labels)} 首，跳过 {len(errors)} 首。特征文件：{FEATURE_FILE}')


if __name__ == '__main__':
    main()
