# 音乐流派分类

## 项目介绍

使用本地 GTZAN 数据集，完成 WAV 音频读取、MFCC 特征提取、KNN 分类、结果评估和单曲预测
数据集来源：Kaggle 上的 GTZAN 副本(https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)，下载解压后将 10 个流派子目录放到 archive/Data/genres_original/ 下即可运行。

## 文件结构

```text
Music_Classification/
├── .venv/                        虚拟环境，依赖装在这里
├── archive/Data/genres_original/   原始音频，按流派分文件夹
└── music_genre_project/
    ├── config.py                 数据路径、类别、随机种子和默认 k
    ├── features.py               单首音频特征提取
    ├── prepare_data.py           遍历数据集并保存特征
    ├── knn.py                    手写 KNN 距离和投票
    ├── train.py                  划分、标准化、训练及评估
    ├── predict.py                单个 WAV 文件预测
    ├── requirements.txt          依赖列表
    ├── README.md                 使用说明
    └── outputs/                  运行后自动生成
```

## 安装和运行

虚拟环境在仓库根目录 `Music_Classification\.venv`
```powershell
cd C:\Users\su\Downloads\Music_Classification\music_genre_project
..\.venv\Scripts\python.exe prepare_data.py
..\.venv\Scripts\python.exe train.py
..\.venv\Scripts\python.exe predict.py "..\archive\Data\genres_original\blues\blues.00000.wav"
```

`..\.venv` 是相对上一行 `cd` 进入的 `music_genre_project` 而言，指向仓库根目录的环境；直接调用解释器即可，不需要激活脚本。特征提取可能需要几分钟，每完成一个类别会打印进度。上面的预测命令只是运行示例，该文件可能属于训练集，不能用这一次预测衡量泛化能力。

换电脑或需要重建环境时，在仓库根目录执行；注意 `-r` 的路径相对当前目录：

```powershell
cd C:\Users\su\Downloads\Music_Classification
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r music_genre_project\requirements.txt
```

如数据集位置变化（仍在 `music_genre_project` 目录下执行）：

```powershell
..\.venv\Scripts\python.exe prepare_data.py --data-dir "D:\数据\genres_original"
```

重新提取特征后应重新运行 train.py。修改 k 可以运行 `train.py --k 3`。默认 k=5；不应反复根据测试集结果选 k，正式调参应另外从训练集划分验证集。

## 方法说明

1. 读取 WAV，双声道取平均转单声道，使用前 30 秒并归一化幅度。不足 1 秒或静音音频会报错。
2. 用 20 毫秒窗口提取 13 个 MFCC 系数，计算每个系数在整段音频上的均值和标准差。
3. 固定随机种子 42，按类别分层划分约 80% 训练数据和 20% 测试数据。
4. 只根据训练集计算各特征的均值和标准差，再标准化两组数据。
5. 对待预测样本计算到每个训练样本的欧氏距离，选最近的 k 个进行投票。平票时优先选择最近邻居中先出现的类别。
6. 输出准确率、各类别 precision / recall / F1 和混淆矩阵。KNN 没有梯度训练过程，保存训练样本就相当于保存模型。

MFCC 计算封装在传统信号处理库中，不需要初学者手写傅里叶变换。建议阅读顺序：config.py → features.py → prepare_data.py → knn.py → train.py → predict.py。

## 输出文件

- features.npz：特征、类别与文件名。
- skipped_files.txt：无法读取的音频及原因；为空表示没有跳过。
- knn_model.npz：仅包含训练集特征、标签、标准化参数和 k。
- metrics.json：准确率、训练和测试数量、k、随机种子。
- classification_report.txt：各类别的评价指标。
- predictions.csv：测试文件名、真实类别、预测类别和是否正确。
- confusion_matrix.png：混淆矩阵，行为真实类别，列为预测类别。

