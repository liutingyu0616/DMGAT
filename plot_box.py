import os
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# plot_dir = [
#     "./PDformer_spy",
#     "./PDformer_two_step",
#     "./PDformer_pu_bagging",
#     "./PDformer_all",
# ]
plot_dir = [
    "./scores",
    "./scores_noDM",
    "./scores_noGCN",
    "./scores_noGAT",
]

data = {
    'AUC': [
        # 第1个模型的5折AUC
        # 第2个模型的5折AUC
    ],
    'AUPR': [
    ]
}

adj_np = pd.read_csv(r"ncrna-drug_split.csv", index_col=0).values

with open(r"fold_info.pickle", "rb") as f:
    fold_info = pickle.load(f)

pos_train_ij_list = fold_info["pos_train_ij_list"]
pos_test_ij_list = fold_info["pos_test_ij_list"]
unlabelled_train_ij_list = fold_info["unlabelled_train_ij_list"]
unlabelled_test_ij_list = fold_info["unlabelled_test_ij_list"]


for dir in plot_dir:
    data['AUC'].append([])
    data['AUPR'].append([])
    for fold in range(5):

        pos_test_ij = pos_test_ij_list[fold]
        unlabelled_test_ij = unlabelled_test_ij_list[fold]

        # test_ij = np.vstack((pos_test_ij, unlabelled_test_ij))
        test_mask_np = np.zeros_like(adj_np)
        test_mask_np[tuple(list(pos_test_ij.T))] = 1
        test_mask_np[tuple(list(unlabelled_test_ij.T))] = 1
        test_ij = np.argwhere(test_mask_np == 1)

        labels = adj_np[tuple(list(test_ij.T))]

        score_path = dir + f"/f{fold}_e199_scores.npy"
        score = np.load(score_path)

        fpr, tpr, thresholds_ = metrics.roc_curve(labels, score)
        auc = metrics.auc(fpr, tpr)
        data['AUC'][-1].append(auc)

        precisions, recalls, thresholds = metrics.precision_recall_curve(labels, score)
        aupr = metrics.auc(recalls, precisions)
        data['AUPR'][-1].append(aupr)



# 生成DataFrame
df_auc = pd.DataFrame({
    'Metric': 'AUC',
    'Value': np.concatenate(data['AUC']),
    'Category': np.repeat(['DMGAT', 'DMGAT(noDM)', 'DMGAT(noGCN)', 'DMGAT(noGAT)'], 5)
})

df_aupr = pd.DataFrame({
    'Metric': 'AUPR',
    'Value': np.concatenate(data['AUPR']),
    'Category': np.repeat(['DMGAT', 'DMGAT(noDM)', 'DMGAT(noGCN)', 'DMGAT(noGAT)'], 5)
})


df = pd.concat([df_auc, df_aupr])

# 设置颜色
colors = ['#9ecae1', '#fdae6b', '#d62728', '#ffeda0']

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制箱线图
sns.boxplot(x='Metric', y='Value', hue='Category', data=df, palette=colors, showfliers=False)

# 设置Y轴范围
plt.ylim(0., 1)

# 调整图例位置
# plt.legend(title='', loc='upper center', ncol=5)
plt.legend(title='')
# 显示图形
plt.show()