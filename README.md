# Bank Marketing 客户响应预测项目

本项目对应《机器学习导论》大作业题目一：使用 UCI Bank Marketing 数据集预测客户是否会订购定期存款。

## 1. 项目目标

根据客户基本信息、金融状态和营销联系信息，建立二分类模型，预测客户是否会订购定期存款，即目标变量 `y` 是否为 `yes`。

## 2. 项目结构

```text
bank_marketing_ml_project/
├── README.md
├── requirements.txt
├── main.py
├── src/
│   ├── data_utils.py
│   ├── train_utils.py
│   └── visualize.py
├── data/                  # 自动下载并解压数据集
└── results/               # 自动保存实验结果、图像、模型
```

## 3. 环境安装

```bash
pip install -r requirements.txt
```

## 4. 运行方式

```bash
python main.py
```

运行后会自动完成：

1. 下载并读取 UCI Bank Marketing 数据集；
2. 进行基本数据分析；
3. 划分训练集和测试集；
4. 对数值特征进行标准化，对类别特征进行 One-Hot 编码；
5. 训练逻辑回归、决策树、随机森林三个模型；
6. 使用交叉验证进行简单调参；
7. 在测试集上评估模型；
8. 保存评估指标、混淆矩阵、ROC 曲线、特征重要性和最优模型。

## 5. 输出文件

运行结束后，`results/` 目录下会生成：

- `metrics.csv`：不同模型的 Accuracy、Precision、Recall、F1、ROC-AUC；
- `classification_report.txt`：最佳模型的分类报告；
- `feature_importance.csv`：随机森林得到的特征重要性排序；
- `best_model.joblib`：保存的最佳模型；
- `figures/target_distribution.png`：目标变量分布图；
- `figures/confusion_matrix.png`：最佳模型混淆矩阵；
- `figures/roc_curve.png`：不同模型 ROC 曲线；
- `figures/top_features.png`：重要特征可视化。

