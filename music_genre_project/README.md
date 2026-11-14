# 音乐流派分类（基础课程版）

## 项目介绍

使用本地 GTZAN 数据集，完成 WAV 音频读取、MFCC 特征提取、KNN 分类、结果评估和单曲预测。只使用基础 Python、NumPy、SciPy、scikit-learn 和 Matplotlib，不使用深度学习框架。KNN 的距离计算和投票由普通函数实现，scikit-learn 仅用于数据划分和评估。

参考教程：https://data-flair.training/blogs/python-project-music-genre-classification/

保留教程的 MFCC + (OUTPUT_DIR / 'skipped_files.txt').write_text('6\n'.join(errors), encoding='utf-8')KNN 思路。本项目为了便于基础课程学习，将教程的均值、协方差距离改成 13 维 MFCC 的均值和标准差，共 26 维特征，再使用标准化后的欧氏距离。因此不是原文算法的逐行复现，结果不能直接与原文比较。KNN 是传统机器学习算法。

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

直接读取上一级的原始数据，不需要复制大体积音频。已检查本地有 10 个类别，每类 100 个 WAV 文件；数量检查不代表所有音频均可解码。类别为 blues、classical、country、disco、hiphop、jazz、metal、pop、reggae、rock。

本项目重新从 WAV 提取特征，不使用已有 CSV：CSV 的特征定义不同，直接混用会造成训练和预测输入不一致。也不将 3 秒切片随机混入训练和测试集，以免同一首歌同时出现在两边。

## 安装和运行

虚拟环境在仓库根目录 `Music_Classification\.venv`，已用 Python 3.13 创建并按 `requirements.txt` 装好依赖。日常运行只需在 PowerShell 中执行：

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

## 当前验证情况与限制

已完成目录、数据数量和依赖安装检查，虚拟环境可用：`features.py` 对 `blues.00000.wav` 的实际调用已跑通，输出 26 维特征。完整的 `prepare_data.py` 和 `train.py` 尚未执行，因此文中不提供未经运行的准确率或模型文件；按上述顺序执行即可生成实际结果。

只支持 SciPy 可读取的 WAV，MP3 需先转换。坏音频会在特征提取阶段跳过并记录，因此实际样本数可能不足 1000。新音频也应是音乐片段；输出是十类中的最近类别，不表示模型能够识别未知流派。整曲随机划分仍不能保证不同艺人或重复录音完全隔离，实验成绩只用于基础练习。
