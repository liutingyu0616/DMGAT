import numpy as np
from utils import *
import torch
from model import *
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pandas as pd
import pickle
import matplotlib
import os
from sklearn.manifold import TSNE
import seaborn as sns
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

sns.set()


matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

seed_everything(42)
device = torch.device("cuda")
path = "../scores/"
if not os.path.exists(path):
    os.makedirs(path)

# load adj, sim
adj_df = pd.read_csv(r"../ncrna-drug_split.csv", index_col=0)
adj_np = adj_df.values
adj_with_sens_np = pd.read_csv(r"../adj_with_sens.csv", index_col=0).values
self_sim = np.load('../self_sim.npy', allow_pickle=True).flat[0]
feat_dm = np.load('../feat_dm.npy', allow_pickle=True).flat[0]
with open(r"fold_info.pickle", "rb") as f:
    fold_info = pickle.load(f)
with open(rf"rn_ij_list.pickle", "rb") as f:
    rn_ij_list = pickle.load(f)

rna_self_sim_np = self_sim['rna_self_sim']
drug_self_sim_np = self_sim['drug_self_sim']

lnc_dmap_np = feat_dm['lnc_dmap']
mi_dmap_np = feat_dm['mi_dmap']
drug_dmap_np = feat_dm['drug_dmap']

lnc_dmap_np = PolynomialFeatures(4).fit_transform(lnc_dmap_np)
mi_dmap_np = PolynomialFeatures(1).fit_transform(mi_dmap_np)
drug_dmap_np = PolynomialFeatures(2).fit_transform(drug_dmap_np)

n_lnc = len(lnc_dmap_np)
n_mi = len(mi_dmap_np)
n_rna = n_lnc + n_mi
n_drug = len(drug_self_sim_np)

diag_mask = rna_self_sim_np!=0

p_gip_list = fold_info["p_gip_list"]
d_gip_list = fold_info["d_gip_list"]

sens_ij_list = np.argwhere(adj_with_sens_np == -1)
sens_ij_df = pd.DataFrame(sens_ij_list)

# rna_self_sim = torch.FloatTensor(rna_self_sim_np).to(device)
# drug_self_sim = torch.FloatTensor(drug_self_sim_np).to(device)
lnc_dmap = torch.FloatTensor(lnc_dmap_np).to(device)
mi_dmap = torch.FloatTensor(mi_dmap_np).to(device)
drug_dmap = torch.FloatTensor(drug_dmap_np).to(device)
adj = torch.FloatTensor(adj_np).to(device)
adj_with_sens = torch.FloatTensor(adj_with_sens_np).to(device)



pos_ij = np.argwhere(adj_with_sens_np == 1)
sens_ij = np.argwhere(adj_with_sens_np == -1)
unlabelled_ij = np.argwhere(adj_with_sens_np == 0)
rn_ij = rn_ij_list[0]

pos_ij_tensor = torch.IntTensor(pos_ij).to(device)
sens_ij_tensor = torch.IntTensor(sens_ij).to(device)
unlabelled_ij_tensor = torch.IntTensor(unlabelled_ij).to(device)
rn_ij_tensor = torch.IntTensor(rn_ij).to(device)

n_heads = 2
linear_out_size = gcn_in_dim = 512
gcn_out_dim = gat_in_dim = 512
gat_hid_dim = 512
gat_out_dim = 512
dropout = 0.
pred_hid_size = 1024

lr, num_epochs = 0.005, 200

class MaskedBCELoss(nn.BCELoss):
    def forward(self, new_p_feat, new_d_feat, adj, train_mask, test_mask):
        self.reduction = "none"
        cosine_sim = F.cosine_similarity(new_p_feat.unsqueeze(1), new_d_feat.unsqueeze(0), dim=2)
        cosine_sim_exp = torch.exp(cosine_sim / 0.5)
        sim_num = adj * cosine_sim_exp * train_mask
        sim_diff = cosine_sim_exp * (1 - adj) * train_mask
        sim_diff_sum = torch.sum(sim_diff, dim=1)
        sim_diff_sum_expend = sim_diff_sum.repeat(new_d_feat.shape[0], 1).T
        sim_den = sim_num + sim_diff_sum_expend
        loss = torch.div(sim_num, sim_den)
        loss1 = torch.clamp(1 - adj+1-train_mask, max=1) + loss
        # loss1 = 1 - adj + loss
        loss_log = -torch.log(loss1)  # 求-log
        loss_c = loss_log.sum()# / (len(torch.nonzero(loss_log)))

        pred = F.sigmoid(new_p_feat.mm(new_d_feat.t()))
        unmasked_loss = super(MaskedBCELoss, self).forward(pred, adj)
        loss_b = (unmasked_loss * train_mask).sum()
        # print(loss_b.item())
        train_loss = loss_b + loss_c
        # train_loss = loss_c
        test_loss = (unmasked_loss * test_mask).sum()
        return train_loss, test_loss



def fit(
    fold_cnt,
    model,
    adj,
    rna_sim,
    drug_sim,
    adj_full,
    lnc_emb, mi_emb, drug_emb,
    train_mask,
    test_mask,
    lr,
    num_epochs,
    pos_test_ij,
    unlabelled_test_ij
):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    model.apply(xavier_init_weights)

    # optimizer = torch.optim.Adam(params ,lr=0)

    # optimizer = torch.optim.RMSprop(model.parameters(), lr, 0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = MaskedBCELoss()

    train_idx = torch.argwhere(train_mask == 1)
    test_idx = torch.argwhere(train_mask == 1)
    # test_idx = torch.argwhere(torch.ones_like(test_mask) == 1)
    for epoch in range(num_epochs):
        # for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        new_p_feat, new_d_feat = model(lnc_emb, mi_emb, drug_emb, rna_sim, drug_sim, adj_full)
        train_loss, test_loss = loss(new_p_feat, new_d_feat, adj, train_mask, test_mask)
        train_loss.backward()
        # grad_clipping(model, 1)
        optimizer.step()

        model.eval()
        new_p_feat, new_d_feat = model(lnc_emb, mi_emb, drug_emb, rna_sim, drug_sim, adj_full)
        pred = F.sigmoid(new_p_feat.mm(new_d_feat.T))
        scores = pred[tuple(list(test_idx.T))].cpu().detach().numpy()

        # print(len(set(scores)))
        # np.save(rf".\scores\f{fold_cnt}_e{epoch}_scores.npy", scores)
        # logger.update(
        #     fold_cnt, epoch, adj, pred, test_idx, train_loss.item(),
        #     test_loss.item(), pos_test_ij, unlabelled_test_ij
        # )


    return pred, new_p_feat, new_d_feat

logger = Logger(5)



# A_corner_np = np.zeros_like(adj_np)
# A_corner_np[tuple(list(pos_train_ij.T))] = 1

# train_mask_np = np.ones_like(adj_np)
train_mask_np = np.zeros_like(adj_np)
train_mask_np[tuple(list(pos_ij.T))] = 1
train_mask_np[tuple(list(sens_ij.T))] = 1
train_mask_np[tuple(list(unlabelled_ij.T))] = 1
train_mask_np[tuple(list(rn_ij.T))] = 1


rna_sim_np = p_gip_list[0]+rna_self_sim_np-p_gip_list[0]*diag_mask*0.5
np.fill_diagonal(rna_sim_np, 1)
drug_sim_np = d_gip_list[0]+drug_self_sim_np
np.fill_diagonal(drug_sim_np, 1)

adj_full_np = np.concatenate(
    (
        np.concatenate((np.eye(len(rna_sim_np)), adj_np), axis=1),
        np.concatenate((adj_np.T, np.eye(len(drug_sim_np))), axis=1),
    ),
    axis=0,
)

rna_sim = torch.FloatTensor(rna_sim_np).to(device)
drug_sim = torch.FloatTensor(drug_sim_np).to(device)
adj_full = torch.FloatTensor(adj_full_np).to(device)

train_mask = torch.FloatTensor(train_mask_np).to(device)
torch.cuda.empty_cache()

linear_layer = Linear(
    lnc_dmap, mi_dmap, drug_dmap, linear_out_size
).to(device)

r_gcn = GCN(
    in_dim=gcn_in_dim,
    out_dim=gcn_out_dim,
    adj=rna_sim
).to(device)
d_gcn = GCN(
    in_dim=gcn_in_dim,
    out_dim=gcn_out_dim,
    adj=drug_sim
).to(device)

gat = GAT(
    in_dim=linear_out_size,
    hid_dim=gat_hid_dim,
    out_dim=gat_out_dim,
    adj_full=adj_full,
    dropout=dropout,
    alpha=0.1,
    nheads=n_heads
).to(device)

predictor = Predictor(gcn_out_dim, pred_hid_size).to(device)

model = PUTransGCN(linear_layer, r_gcn, d_gcn, gat, predictor).to(device)
pred = fit(
    0,
    model,
    adj,
    rna_sim,
    drug_sim,
    adj_full,
    lnc_dmap, mi_dmap, drug_dmap,
    train_mask,
    train_mask,
    lr,
    num_epochs,
    pos_ij_tensor,
    unlabelled_ij_tensor
)
pred, new_p_feat, new_d_feat = pred[0], pred[1], pred[2]
unlabelled_ij = np.argwhere(adj_np != 1)

scores = pred[tuple(list(unlabelled_ij.T))].cpu().detach().numpy()

pred_np = pred.cpu().detach().numpy()
new_p_feat_np = new_p_feat.cpu().detach().numpy()
new_d_feat_np = new_d_feat.cpu().detach().numpy()

pred_np_copy = pred_np.copy()
pred_np_copy[tuple(list(pos_ij.T))] = 0
flat_array = pred_np_copy.flatten()
sorted_indices = np.argsort(-flat_array)
rn_indices = sorted_indices[:500]
rn_positions = np.unravel_index(rn_indices, pred_np_copy.shape)

pred_top = np.zeros_like(adj_np)
pred_top[rn_positions] = 1
rnd_mat = (np.random.rand(*adj_np.shape)<(len(pos_ij)/np.multiply(*adj_np.shape))).astype(int)

rna_sim_list = []
for i in range(adj_with_sens_np.shape[1]):
    if( (adj_with_sens_np[:, i] == 1).sum()) !=0:
        rna_pos_sim = cosine_similarity(new_p_feat_np[adj_with_sens_np[:, i] == 1]).mean()
        rna_sim_list.append(rna_pos_sim)

rna_sim_top_list = []
for i in range(adj_with_sens_np.shape[1]):
    if( (pred_top[:, i] == 1).sum()) !=0:
        rna_pos_sim = cosine_similarity(new_p_feat_np[pred_top[:, i] == 1]).mean()
        rna_sim_top_list.append(rna_pos_sim)

rna_sim_rnd_list = []
for i in range(adj_with_sens_np.shape[1]):
    if ((rnd_mat[:, i] == 1).sum()) != 0:
        rna_pos_sim = cosine_similarity(new_p_feat_np[rnd_mat[:, i] == 1]).mean()
        rna_sim_rnd_list.append(rna_pos_sim)

fig, ax = plt.subplots()
data = [rna_sim_list, rna_sim_top_list, rna_sim_rnd_list]
ax.boxplot(data, showfliers=False)

plt.ylim(-0.05,1.05)
plt.title('Distribution of the mean of cosine similarity\nbetween ncRNAs associated with the same drug')
ax.set_xticklabels(["Known association", "Predict association", "Random association"])
plt.savefig('ncRNAs_sim.tiff', dpi=300)
plt.show()

drug_sim_list = []
for i in range(adj_with_sens_np.shape[0]):
    if( (adj_with_sens_np[i, :] == 1).sum()) !=0:
        drug_pos_sim = cosine_similarity(new_d_feat_np[adj_with_sens_np[i, :] == 1]).mean()
        drug_sim_list.append(drug_pos_sim)

drug_sim_top_list = []
for i in range(adj_with_sens_np.shape[0]):
    if( (pred_top[i, :] == 1).sum()) !=0:
        drug_pos_sim = cosine_similarity(new_d_feat_np[pred_top[i, :] == 1]).mean()
        drug_sim_top_list.append(drug_pos_sim)

drug_sim_rnd_list = []
for i in range(adj_with_sens_np.shape[0]):
    if ((rnd_mat[i, :] == 1).sum()) != 0:
        drug_pos_sim = cosine_similarity(new_d_feat_np[rnd_mat[i, :] == 1]).mean()
        drug_sim_rnd_list.append(drug_pos_sim)

fig, ax = plt.subplots()
d_data = [drug_sim_list, drug_sim_top_list, drug_sim_rnd_list]
plt.title('Distribution of the mean of cosine similarity\nbetween drugs associated with the same ncRNA ')
ax.boxplot(d_data)
plt.ylim(-0.05,1.05)
ax.set_xticklabels(["Known association", "Predict association", "Random association"])
plt.savefig('drugs_sim.tiff', dpi=300)
plt.show()


# plot_xy1(std_data, cluster, label,links,"t-SNE")

# adj_df_copy = adj_df.copy()
# adj_df_copy.index = [f"{idx}({i})" for i, idx in enumerate(adj_df_copy.index)]
# adj_df_copy.columns  = [f"{idx}({i})" for i, idx in enumerate(adj_df_copy.columns )]

# from umap import UMAP
# from sklearn.decomposition import PCA
#
# df = pd.DataFrame(new_feat_np, columns=['x', 'y'])
# df['cluster'] = cluster
# df['label'] = label
#
# umap_2d = UMAP(n_components=2, init='random', random_state=0)
# proj_2d = umap_2d.fit_transform(new_feat_np)
# plot_xy(proj_2d, cluster, label,"t-sne")
#
# pca = PCA(n_components=2)
# pca_2d = pca.fit_transform(new_feat_np)
# plot_xy(pca_2d, cluster, label,"t-sne")


a=1
# 创建空的DataFrame
# piRNA_name = adj_df.index
# disease_name = adj_df.columns
# scores_list = []
#
# for i, index in enumerate(unlabelled_ij):
#     score = scores[i]
#     piRNA = piRNA_name[index[0]]
#     disease = disease_name[index[1]]
#     scores_list.append([piRNA, disease, score])
#
# scores_df = pd.DataFrame(scores_list, columns=['piRNA', 'disease', 'score'])
#
# scores_df = scores_df.sort_values(by='score', ascending=False)
#
# scores_df.to_csv('unlabelled_scores.csv', index=False)
# logger.save("DMGAT")


# torch.save(model, "params.pt")