import numpy as np
from utils import *
import torch
from model import *
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import pickle
import matplotlib
import os

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

seed_everything(42)
device = torch.device("cuda")


# load adj, sim
adj_np = pd.read_csv(r"ncrna-drug_split.csv", index_col=0).values
adj_with_sens_np = pd.read_csv(r"adj_with_sens.csv", index_col=0).values
self_sim = np.load('self_sim.npy', allow_pickle=True).flat[0]
feat_dm = np.load('feat_2d.npy', allow_pickle=True).flat[0]
with open(r"fold_info.pickle", "rb") as f:
    fold_info = pickle.load(f)
with open(rf"rn_ij_list.pickle", "rb") as f:
    rn_ij_list = pickle.load(f)

rna_self_sim_np = self_sim['rna_self_sim']
drug_self_sim_np = self_sim['drug_self_sim']

lnc_dmap_np = feat_dm['lnc_emb_2d']
mi_dmap_np = feat_dm['mi_emb_2d']
drug_dmap_np = feat_dm['drug_emb_2d']
# lnc_dmap_np = feat_dm['lnc_dmap']
# mi_dmap_np = feat_dm['mi_dmap']
# drug_dmap_np = feat_dm['drug_dmap']


lnc_dmap_mean = lnc_dmap_np.mean(axis=0)
mi_dmap_mean = mi_dmap_np.mean(axis=0)
drug_dmap_mean = drug_dmap_np.mean(axis=0)

rna_seq = pd.read_csv(r"rna_seq.csv")
drug_seq=pd.read_csv('drug_smiles.csv')

lnc_seq = rna_seq[rna_seq['type']=='lncRNA']
mi_seq = rna_seq[rna_seq['type']=='miRNA']

lnc_not_found_indices = np.where(lnc_seq['seq'] == 'NotFound')[0]
mi_not_found_indices = np.where(mi_seq['seq'] == 'NotFound')[0]
drug_not_found_indices = np.where(drug_seq['smiles'] == 'NotFound')[0]

for idx in lnc_not_found_indices:
    lnc_dmap_np = np.insert(lnc_dmap_np, idx, lnc_dmap_mean, axis=0)

for idx in mi_not_found_indices:
    mi_dmap_np = np.insert(mi_dmap_np, idx, mi_dmap_mean, axis=0)

for idx in drug_not_found_indices:
    drug_dmap_np = np.insert(drug_dmap_np, idx, drug_dmap_mean, axis=0)

# lnc_dmap_np = PolynomialFeatures(4).fit_transform(lnc_dmap_np)
# mi_dmap_np = PolynomialFeatures(1).fit_transform(mi_dmap_np)
# drug_dmap_np = PolynomialFeatures(2).fit_transform(drug_dmap_np)

n_circ = 2
n_lnc = len(lnc_dmap_np)
n_mi = len(mi_dmap_np)
n_pi = 1
n_rna = n_circ + n_lnc + n_mi + n_pi
n_drug = len(drug_self_sim_np)

diag_mask = rna_self_sim_np!=0
diag_mask[:2, :2] = True

pos_train_ij_list = fold_info["pos_train_ij_list"]
pos_test_ij_list = fold_info["pos_test_ij_list"]
unlabelled_train_ij_list = fold_info["unlabelled_train_ij_list"]
unlabelled_test_ij_list = fold_info["unlabelled_test_ij_list"]
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

n_heads = 2
linear_out_size = gcn_in_dim = 512
gcn_out_dim = gat_in_dim = 512
gat_hid_dim = 512
gat_out_dim = 512
dropout = 0.21
pred_hid_size = 1024

lr, num_epochs = 0.01, 200

class MaskedBCELoss(nn.BCELoss):
    def forward(self, pred, adj, train_mask, test_mask):
        self.reduction = "none"
        unweighted_loss = super(MaskedBCELoss, self).forward(pred, adj)
        train_loss = (unweighted_loss * train_mask).sum()
        test_loss = (unweighted_loss * test_mask).sum()
        return train_loss, test_loss
# class MaskedMSELoss(nn.MSELoss):
#     def forward(self, pred, adj, train_mask, test_mask):
#         self.reduction = "none"
#         unweighted_loss = super(MaskedMSELoss, self).forward(pred, adj)
#         train_loss = (unweighted_loss * train_mask).sum()
#         test_loss = (unweighted_loss * test_mask).sum()
#         return train_loss, test_loss


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


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
    test_idx = torch.argwhere(test_mask == 1)
    # test_idx = torch.argwhere(torch.ones_like(test_mask) == 1)
    for epoch in range(num_epochs):
        # for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        pred = model(lnc_emb, mi_emb, drug_emb, rna_sim, drug_sim, adj_full)
        train_loss, test_loss = loss(pred, adj, train_mask, test_mask)
        # train_loss.requires_grad_(True)
        train_loss.backward()
        # grad_clipping(model, 1)
        optimizer.step()

        model.eval()
        pred = model(lnc_emb, mi_emb, drug_emb, rna_sim, drug_sim, adj_full)

        scores = pred[tuple(list(test_idx.T))].cpu().detach().numpy()
        # print(len(set(scores)))
        np.save(rf".\scores_noDM\f{fold_cnt}_e{epoch}_scores.npy", scores)
        logger.update(
            fold_cnt, epoch, adj, pred, test_idx, train_loss.item(), test_loss.item()
        )

    return 0

logger = Logger(5)

for i in range(5):
    print(f"fold {i}")
    pos_train_ij = pos_train_ij_list[i]
    pos_test_ij = pos_test_ij_list[i]
    unlabelled_train_ij = unlabelled_train_ij_list[i]
    unlabelled_test_ij = unlabelled_test_ij_list[i]
    rn_ij = rn_ij_list[i]

    # A_corner_np = np.zeros_like(adj_np)
    # A_corner_np[tuple(list(pos_train_ij.T))] = 1

    train_mask_np = np.ones_like(adj_np)
    # train_mask_np = np.zeros_like(adj_np)
    train_mask_np[tuple(list(pos_train_ij.T))] = 1

    unlabelled_train_ij_df = pd.DataFrame(unlabelled_train_ij)
    sens_train_ij = pd.merge(sens_ij_df, unlabelled_train_ij_df).values
    # train_mask_np[tuple(list(unlabelled_train_ij.T))] = 1
    train_mask_np[tuple(list(sens_train_ij.T))] = 1
    train_mask_np[tuple(list(rn_ij.T))] = 1

    train_label_np = train_mask_np * adj_np

    test_mask_np = np.zeros_like(adj_np)
    test_mask_np[tuple(list(pos_test_ij.T))] = 1
    test_mask_np[tuple(list(unlabelled_test_ij.T))] = 1
    test_label_np = test_mask_np*adj_np

    rna_sim_np = p_gip_list[i]+rna_self_sim_np-p_gip_list[i]*diag_mask*0.5
    np.fill_diagonal(rna_sim_np, 1)
    drug_sim_np = d_gip_list[i]+drug_self_sim_np
    np.fill_diagonal(drug_sim_np, 1)

    adj_full_np = np.concatenate(
        (
            np.concatenate((np.eye(len(rna_sim_np)), train_label_np), axis=1),
            np.concatenate((train_label_np.T, np.eye(len(drug_sim_np))), axis=1),
        ),
        axis=0,
    )

    rna_sim = torch.FloatTensor(rna_sim_np).to(device)
    drug_sim = torch.FloatTensor(drug_sim_np).to(device)
    adj_full = torch.FloatTensor(adj_full_np).to(device)

    train_mask = torch.FloatTensor(train_mask_np).to(device)
    test_mask = torch.FloatTensor(test_mask_np).to(device)
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
    fit(
        i,
        model,
        adj,
        rna_sim,
        drug_sim,
        adj_full,
        lnc_dmap, mi_dmap, drug_dmap,
        train_mask,
        test_mask,
        lr,
        num_epochs,
    )
    max_allocated_memory = torch.cuda.max_memory_allocated()
    print(f"最大已分配内存量: {max_allocated_memory / 1024 ** 2} MB")

# logger.save("DMGAT_noDM")
# torch.save(model, "params.pt")