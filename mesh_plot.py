import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

mesh = np.load('mesh.npy', allow_pickle=True).flat[0]

offset = 0.88

auc_grid = mesh['auc_grid']-offset
aupr_grid = mesh['aupr_grid']-offset

auc_grid[auc_grid<0]=0
aupr_grid[aupr_grid<0]=0

print(auc_grid)
print(aupr_grid)

# auc_grid = np.arange(25).reshape(5,5)

colors = np.full(auc_grid.shape, 'skyblue', dtype=object)  # 先全部设为蓝色
colors[1, 3] = 'r'  # 修改第 2 行（索引 1），第 4 列（索引 3）的颜色为红色

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 6))

# 第一个子图 - AUC
ax1 = fig.add_subplot(121, projection='3d')
x, y = np.meshgrid(np.arange(auc_grid.shape[0]), np.arange(auc_grid.shape[1]))
ax1.bar3d(x.flatten()+0.75, y.flatten()+0.75, np.zeros_like(auc_grid.flatten())+offset,
          0.5, 0.5, auc_grid.flatten(), shade=True, zsort='average', color=colors.flatten())
ax1.set_xlabel('GAT layer')
ax1.set_ylabel('GCN layer')
ax1.set_title('(a) AUC')
ax1.set_xlim(0.5, 5.5)
ax1.set_ylim(0.5, 5.5)
plt.tight_layout()

# 第二个子图 - AUPR
ax2 = fig.add_subplot(122, projection='3d')
x, y = np.meshgrid(np.arange(aupr_grid.shape[0]), np.arange(aupr_grid.shape[1]))
ax2.bar3d(x.flatten()+0.75, y.flatten()+0.75, np.zeros_like(aupr_grid.flatten())+offset,
          0.5, 0.5, aupr_grid.flatten(), shade=True, zsort='average', color=colors.flatten())
ax2.set_xlabel('GAT layer')
ax2.set_ylabel('GCN layer')
ax2.set_title('(b) AUPR')
ax2.set_xlim(0.5, 5.5)
ax2.set_ylim(0.5, 5.5)
plt.tight_layout()
# 保存 & 显示
plt.savefig('auc_aupr_mesh.tiff', dpi=300)
plt.show()

#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x, y = np.meshgrid(np.arange(auc_grid.shape[0]), np.arange(auc_grid.shape[1]))
# ax.bar3d(x.flatten()+0.75, y.flatten()+0.75, np.zeros_like(auc_grid.flatten())+offset, 0.5, 0.5, auc_grid.flatten(), shade=True,
#         zsort='average',color=colors.flatten())
# ax.set_xlabel('GAT layer')
# ax.set_ylabel('GCN layer')
# plt.title('AUC')
# plt.xlim(0.5, 5.5)
# plt.ylim(0.5, 5.5)
# plt.savefig('auc_mesh.tiff', dpi=300)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x, y = np.meshgrid(np.arange(aupr_grid.shape[0]), np.arange(aupr_grid.shape[1]))
# ax.bar3d(x.flatten()+0.75, y.flatten()+0.75, np.zeros_like(aupr_grid.flatten())+offset, 0.5, 0.5, aupr_grid.flatten(), shade=True,
#         zsort='average',color=colors.flatten())
# ax.set_xlabel('GAT layer')
# ax.set_ylabel('GCN layer')
# plt.title('AUPR')
# plt.xlim(0.5, 5.5)
# plt.ylim(0.5, 5.5)
# plt.savefig('aupr_mesh.tiff', dpi=300)
# plt.show()

a=1