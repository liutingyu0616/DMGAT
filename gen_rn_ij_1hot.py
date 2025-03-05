import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

adj_with_sens_np = pd.read_csv(r"adj_with_sens.csv", index_col=0).values

feat_dm = np.load('feat_1hot.npy', allow_pickle=True).flat[0]
lnc_dmap_np = feat_dm['lnc_dmap']
mi_dmap_np = feat_dm['mi_dmap']
drug_dmap_np = feat_dm['drug_dmap']

num_p, num_d = adj_with_sens_np.shape
n_lnc = len(lnc_dmap_np)
n_mi = len(mi_dmap_np)
n_rna = n_lnc + n_mi

feat_mat = np.zeros((num_p, num_d, (mi_dmap_np.shape[1] + drug_dmap_np.shape[1])))
for i in range(n_mi):
    for j in range(num_d):
        feat_mat[i+n_lnc, j] = np.append(mi_dmap_np[i], drug_dmap_np[j])

with open(r"fold_info.pickle", "rb") as f:
    fold_info = pickle.load(f)

pos_train_ij_list = fold_info["pos_train_ij_list"]
pos_test_ij_list = fold_info["pos_test_ij_list"]
unlabelled_train_ij_list = fold_info["unlabelled_train_ij_list"]
unlabelled_test_ij_list = fold_info["unlabelled_test_ij_list"]

# load adj, sim
adj_np = pd.read_csv(r"ncrna-drug_split.csv", index_col=0).values
adj_with_sens_np = pd.read_csv(r"adj_with_sens.csv", index_col=0).values
sens_ij = np.argwhere(adj_with_sens_np == -1)
sens_ij = sens_ij[(sens_ij[:, 0] > n_lnc) & (sens_ij[:, 0] < n_lnc + n_mi)]
sens_ij_df = pd.DataFrame(sens_ij)
rn_ij_list = []
for i in range(5):
    print(f"fold {i}")
    pos_train_ij = pos_train_ij_list[i]
    pos_test_ij = pos_test_ij_list[i]
    unlabelled_train_ij = unlabelled_train_ij_list[i]
    unlabelled_test_ij = unlabelled_test_ij_list[i]

    pos_train_ij = pos_train_ij[(pos_train_ij[:, 0] > n_lnc) & (pos_train_ij[:, 0] < n_lnc + n_mi)]
    pos_test_ij = pos_test_ij[(pos_test_ij[:, 0] > n_lnc) & (pos_test_ij[:, 0] < n_lnc + n_mi)]
    unlabelled_train_ij = unlabelled_train_ij[(unlabelled_train_ij[:, 0] > n_lnc) & (unlabelled_train_ij[:, 0] < n_lnc + n_mi)]
    unlabelled_test_ij = unlabelled_test_ij[(unlabelled_test_ij[:, 0] > n_lnc) & (unlabelled_test_ij[:, 0] < n_lnc + n_mi)]

    train_ij = np.vstack((pos_train_ij, unlabelled_train_ij))

    unlabelled_train_ij_df = pd.DataFrame(unlabelled_train_ij)
    sens_train_ij = pd.merge(sens_ij_df, unlabelled_train_ij_df).values

    # rf_pred_ij = unlabelled_train_ij_df[~unlabelled_train_ij_df.isin(sens_ij_df).all(axis=1)].values

    result = unlabelled_train_ij_df.merge(sens_ij_df,  how='left', indicator=True)
    rf_pred_ij = result[result['_merge'] == 'left_only'].drop(columns=['_merge']).values

    pred_feat = feat_mat[tuple(list(rf_pred_ij.T))]
    # unlabelled_train_ij_df[~unlabelled_train_ij_df.isin(pd.DataFrame(sens_train_ij))].dropna()
    pos_ij_folds = np.array_split(pos_train_ij, 7)

    sens_feat = feat_mat[tuple(list(sens_ij.T))]
    prob_mat = np.ones_like(adj_np)*7.
    prob_mat[tuple(list(rf_pred_ij.T))] = 0

    for j in range(7):
        pos_train_1fold_ij = pos_ij_folds[j]
        rf_train_ij = np.vstack((pos_train_1fold_ij, sens_train_ij))

        train_feat = feat_mat[tuple(list(rf_train_ij.T))]
        rf_train_label = adj_with_sens_np[tuple(list(rf_train_ij.T))]
        regressor = RandomForestClassifier(n_jobs=-1, random_state=42)
        regressor.fit(train_feat, rf_train_label)

        pred_prob = regressor.predict_proba(pred_feat)[:, 1]
        prob_mat[tuple(list(rf_pred_ij.T))] += pred_prob

    flat_array = prob_mat.flatten()
    sorted_indices = np.argsort(flat_array)  # 负号表示从大到小排序
    rn_indices = sorted_indices[:len(pos_train_ij)-len(sens_train_ij)]
    rn_positions = np.unravel_index(rn_indices, prob_mat.shape)
    # sorted_elements = flat_array[rn_indices]
    rn_ij = np.vstack((rn_positions[0], rn_positions[1])).T
    rn_ij = np.vstack((rn_ij, sens_train_ij))
    rn_ij_list.append(rn_ij)

with open("rn_ij_list_1hot.pickle", "wb") as f:
    pickle.dump(rn_ij_list, f)
a=1