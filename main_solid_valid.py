import numpy as np
from utils_solid import *
import torch
from model_multi_layers import *
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import pickle
import matplotlib
import os

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

seed_everything(42)
device = torch.device("cuda")
path = "scores/"
if not os.path.exists(path):
    os.makedirs(path)

# load adj, sim
adj_np = pd.read_csv(r"ncrna-drug_split.csv", index_col=0).values
adj_with_sens_np = pd.read_csv(r"adj_with_sens.csv", index_col=0).values
self_sim = np.load('self_sim.npy', allow_pickle=True).flat[0]
feat_dm = np.load('feat_dm.npy', allow_pickle=True).flat[0]
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

pos_train_ij_list = fold_info["pos_train_ij_list"]
pos_test_ij_list = fold_info["pos_test_ij_list"]
unlabelled_train_ij_list = fold_info["unlabelled_train_ij_list"]
unlabelled_test_ij_list = fold_info["unlabelled_test_ij_list"]
p_gip_list = fold_info["p_gip_list"]
d_gip_list = fold_info["d_gip_list"]

sens_ij = np.argwhere(adj_with_sens_np == -1)
sens_ij_df = pd.DataFrame(sens_ij)

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
dropout = 0.6
pred_hid_size = 1024

lr, num_epochs = 0.005, 200

# class MaskedBCELoss(nn.BCELoss):
#     def forward(self, pred, adj, train_mask, test_mask):
#         self.reduction = "none"
#         unweighted_loss = super(MaskedBCELoss, self).forward(pred, adj)
#         train_loss = (unweighted_loss * train_mask).sum()
#         test_loss = (unweighted_loss * test_mask).sum()
#         return train_loss, test_loss

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
        train_loss = (unmasked_loss * train_mask).sum()
        test_loss = (unmasked_loss * test_mask).sum()

        return train_loss, test_loss

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
    pos_test_ij,
    unlabelled_test_ij,
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
        new_p_feat, new_d_feat = model(lnc_emb, mi_emb, drug_emb, rna_sim, drug_sim, adj_full)
        train_loss, test_loss = loss(new_p_feat, new_d_feat, adj, train_mask, test_mask)
        # train_loss.requires_grad_(True)
        train_loss.backward()
        # grad_clipping(model, 1)
        optimizer.step()

        model.eval()
        new_p_feat, new_d_feat = model(lnc_emb, mi_emb, drug_emb, rna_sim, drug_sim, adj_full)
        pred = F.sigmoid(new_p_feat.mm(new_d_feat.T))
        scores = pred[tuple(list(test_idx.T))].cpu().detach().numpy()
        # print(len(set(scores)))
        # np.save(rf".\scores\f{fold_cnt}_e{epoch}_scores.npy", scores)
        logger.update(
            fold_cnt, epoch, adj, pred, test_idx, train_loss.item(),
            test_loss.item(), pos_test_ij, unlabelled_test_ij
        )

    return 0

logger = Logger(5)

for i in range(5):
    print(f"fold {i}")
    pos_train_ij = pos_train_ij_list[i]
    pos_test_ij = pos_test_ij_list[i]
    unlabelled_train_ij = unlabelled_train_ij_list[i]
    unlabelled_test_ij = unlabelled_test_ij_list[i]
    np.random.shuffle(unlabelled_test_ij)
    rn_ij = rn_ij_list[i]
    sens_test_ij = pd.merge(sens_ij_df, pd.DataFrame(unlabelled_test_ij)).values

    # A_corner_np = np.zeros_like(adj_np)
    # A_corner_np[tuple(list(pos_train_ij.T))] = 1

    # train_mask_np = np.ones_like(adj_np)
    train_mask_np = np.zeros_like(adj_np)
    train_mask_np[tuple(list(pos_train_ij.T))] = 1
    train_mask_np[tuple(list(sens_ij.T))] = 1
    train_mask_np[tuple(list(rn_ij.T))] = 1
    train_label_np = train_mask_np * adj_np

    test_mask_np = np.zeros_like(adj_np)
    test_mask_np[tuple(list(pos_test_ij.T))] = 1
    test_mask_np[tuple(list(sens_test_ij.T))] = 1
    # test_mask_np[tuple(list(unlabelled_test_ij.T))] = 1
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
    pos_test_ij_tensor = torch.IntTensor(pos_test_ij).to(device)
    unlabelled_test_ij_tensor = torch.IntTensor(unlabelled_test_ij).to(device)
    torch.cuda.empty_cache()

    linear_layer = Linear(
        lnc_dmap, mi_dmap, drug_dmap, linear_out_size
    ).to(device)

    r_gcn_list = [GCN(
        in_dim=gcn_in_dim,
        out_dim=gcn_out_dim,
        adj=rna_sim
    ).to(device) for _ in range(2)]

    d_gcn_list = [GCN(
        in_dim=gcn_in_dim,
        out_dim=gcn_out_dim,
        adj=drug_sim
    ).to(device) for _ in range(2)]

    gat_list = [GAT(
        in_dim=linear_out_size,
        hid_dim=gat_hid_dim,
        out_dim=gat_out_dim,
        adj_full=adj_full,
        dropout=dropout,
        alpha=0.1,
        nheads=n_heads
    ).to(device) for _ in range(4)]

    predictor = Predictor(gcn_out_dim, pred_hid_size).to(device)

    model = PUTransGCN(linear_layer, r_gcn_list, d_gcn_list, gat_list, predictor).to(device)
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
        pos_test_ij_tensor,
        unlabelled_test_ij_tensor,
    )
    max_allocated_memory = torch.cuda.max_memory_allocated()
    print(f"最大已分配内存量: {max_allocated_memory / 1024 ** 2} MB")

logger.save("DMGAT_solid_valid")
# torch.save(model, "params.pt")