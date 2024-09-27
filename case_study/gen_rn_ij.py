import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

adj_with_sens_np = pd.read_csv(r"../adj_with_sens.csv", index_col=0).values

feat_dm = np.load('../feat_dm.npy', allow_pickle=True).flat[0]
lnc_dmap_np = feat_dm['lnc_dmap']
mi_dmap_np = feat_dm['mi_dmap']
drug_dmap_np = feat_dm['drug_dmap']

num_p, num_d = adj_with_sens_np.shape
n_circ = 2
n_lnc = len(lnc_dmap_np)
n_mi = len(mi_dmap_np)
n_pi = 1
n_rna = n_circ + n_lnc + n_mi + n_pi

feat_mat = np.zeros((num_p, num_d, (mi_dmap_np.shape[1] + drug_dmap_np.shape[1])))
for i in range(n_mi):
    for j in range(num_d):
        feat_mat[i+n_circ+n_lnc, j] = np.append(mi_dmap_np[i], drug_dmap_np[j])


# load adj, sim
adj_np = pd.read_csv(r"../ncrna-drug_split.csv", index_col=0).values
pos_ij = np.argwhere(adj_with_sens_np == 1)
unlabelled_ij = np.argwhere(adj_with_sens_np == 0)
sens_ij = np.argwhere(adj_with_sens_np == -1)
pos_ij = pos_ij[(pos_ij[:, 0] > n_circ + n_lnc) & (pos_ij[:, 0] < n_circ + n_lnc + n_mi)]
unlabelled_ij = unlabelled_ij[(unlabelled_ij[:, 0] > n_circ + n_lnc) & (unlabelled_ij[:, 0] < n_circ + n_lnc + n_mi)]
sens_ij = sens_ij[(sens_ij[:, 0] > n_circ + n_lnc) & (sens_ij[:, 0] < n_circ + n_lnc + n_mi)]

sens_ij_df = pd.DataFrame(sens_ij)
rn_ij_list = []

pos_ij_folds = np.array_split(pos_ij, 7)
pred_feat = feat_mat[tuple(list(unlabelled_ij.T))]
prob_mat = np.ones_like(adj_np) * 7.

for j in range(7):
    pos_train_1fold_ij = pos_ij_folds[j]
    rf_train_ij = np.vstack((pos_train_1fold_ij, sens_ij))

    train_feat = feat_mat[tuple(list(rf_train_ij.T))]
    rf_train_label = adj_with_sens_np[tuple(list(rf_train_ij.T))]
    regressor = RandomForestClassifier(n_jobs=-1, random_state=42)
    regressor.fit(train_feat, rf_train_label)

    pred_prob = regressor.predict_proba(pred_feat)[:, 1]
    prob_mat[tuple(list(unlabelled_ij.T))] += pred_prob

flat_array = prob_mat.flatten()
sorted_indices = np.argsort(flat_array)  # 负号表示从大到小排序
rn_indices = sorted_indices[:len(pos_ij)-len(sens_ij)]
rn_positions = np.unravel_index(rn_indices, prob_mat.shape)
# sorted_elements = flat_array[rn_indices]
rn_ij = np.vstack((rn_positions[0], rn_positions[1])).T

rn_ij_list.append(rn_ij)

with open("rn_ij_list.pickle", "wb") as f:
    pickle.dump(rn_ij_list, f)
a=1