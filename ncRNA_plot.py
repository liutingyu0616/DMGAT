import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['AUC', 'AUPR', 'Recall', 'F1-score']
DMGAT_no_diffusion_map = [0.8464, 0.8563, 0.9527, 0.7799]
DMGAT_no_GCN = [0.8270, 0.8416, 0.9636, 0.7618]
DMGAT_no_GAT = [0.8316, 0.8661, 0.9584, 0.7874]
DMGAT = [0.8921, 0.8938, 0.9562, 0.8270]

# 将数据转换为numpy数组
values = np.array([DMGAT_no_diffusion_map, DMGAT_no_GCN, DMGAT_no_GAT, DMGAT])

# 角度
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

# 闭合图形
values = np.concatenate((values, values[:,[0]]), axis=1)
angles += angles[:1]

# 绘制雷达图
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# 绘制每个模型的线和点
ax.plot(angles, values[0], color='red', alpha=0.8, linewidth=4, linestyle='solid', marker='o', markersize=12, label='DMGAT (no diffusion map)')
ax.plot(angles, values[1], color='blue', alpha=0.8, linewidth=4, linestyle='solid', marker='o', markersize=12, label='DMGAT (no GCN)')
ax.plot(angles, values[2], color='green', alpha=0.8, linewidth=4, linestyle='solid', marker='o', markersize=12, label='DMGAT (no GAT)')
ax.plot(angles, values[3], color='orange', alpha=0.8, linewidth=4, linestyle='solid', marker='o', markersize=12, label='DMGAT')

# 添加标签
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# 将标签旋转45度，使其更易读
# for label in ax.get_xticklabels():
#     label.set_rotation(45)
#     label.set_horizontalalignment('right')

# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(0.8, 1.8))
# plt.bar_label()
# 显示图形
plt.tight_layout()
plt.savefig('radar.tiff', dpi=350)
plt.show()