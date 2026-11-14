from pathlib import Path

# 路径以代码所在文件夹为准，方便从不同目录运行。
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR.parent / 'archive' / 'Data' / 'genres_original'
OUTPUT_DIR = PROJECT_DIR / 'outputs'
FEATURE_FILE = OUTPUT_DIR / 'features.npz'
MODEL_FILE = OUTPUT_DIR / 'knn_model.npz'
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']
RANDOM_SEED = 42
K = 5
