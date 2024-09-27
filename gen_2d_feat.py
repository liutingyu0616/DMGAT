import numpy as np
import torch
from torch import nn


def pad_columns(array, target_columns):
    # 获取原数组的形状
    rows, cols = array.shape
    # 如果当前列数已经大于等于目标列数，则返回原数组
    if cols >= target_columns:
        return array
    # 计算需要填充的列数
    padding_columns = target_columns - cols
    # 使用 np.pad 函数进行填充
    padded_array = np.pad(array, ((0, 0), (0, padding_columns)), mode='constant', constant_values=0)
    return padded_array

gensim_feat_lnc_128 = np.load('gensim_feat_lnc_128.npy', allow_pickle=True).flat[0]
gensim_feat_mi_128 = np.load('gensim_feat_mi_128.npy', allow_pickle=True).flat[0]
gensim_feat_drug_128 = np.load('gensim_feat_drug_128.npy', allow_pickle=True).flat[0]

lnc_kmers_emb = gensim_feat_lnc_128["kmers_emb"]
lnc_pad_kmers_id_seq = gensim_feat_lnc_128["pad_kmers_id_seq"][:,:150]
mi_kmers_emb = gensim_feat_mi_128["kmers_emb"]
mi_pad_kmers_id_seq = gensim_feat_mi_128["pad_kmers_id_seq"]
drug_kmers_emb = gensim_feat_drug_128["kmers_emb"]
drug_pad_kmers_id_seq = gensim_feat_drug_128["pad_kmers_id_seq"]


lnc_emb = lnc_kmers_emb[lnc_pad_kmers_id_seq]
mi_emb = mi_kmers_emb[mi_pad_kmers_id_seq]
drug_emb = drug_kmers_emb[drug_pad_kmers_id_seq]

lnc_emb = torch.tensor(lnc_emb)
mi_emb = torch.tensor(mi_emb)
drug_emb = torch.tensor(drug_emb)


class Model(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.emb = emb
        self.k = nn.Parameter(torch.empty(size=(emb.shape[-1], 1)))
        torch.nn.init.xavier_normal_(self.k)

    def forward(self):
        out = torch.squeeze(self.emb @ self.k)
        return out


def cal_loss(out, model):
    out = out - out.mean(axis=0)
    norm = torch.sqrt(sum(model.k * model.k))
    loss = -sum(out.std(axis=0)) + 10000000 * (norm - 1).abs()
    return loss, norm



def fit(emb):
    model = Model(emb)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
        optimizer.zero_grad()
        out = model()
        loss, norm = cal_loss(out, model)
        loss.backward()
        optimizer.step()
        print('epoch', epoch, 'norm', norm.detach().numpy())
    return out.detach().numpy(), model.k.detach().numpy()

feat_2d = {}
lnc_emb_2d, lnc_param = fit(lnc_emb)
mi_emb_2d, mi_param = fit(mi_emb)
drug_emb_2d, drug_param = fit(drug_emb)

feat_2d['lnc_emb_2d'] = lnc_emb_2d
feat_2d['lnc_param'] = lnc_param
feat_2d['mi_emb_2d'] = mi_emb_2d
feat_2d['mi_param'] = mi_param
feat_2d['drug_emb_2d'] = drug_emb_2d
feat_2d['drug_param'] = drug_param

with open('feat_2d.npy', 'wb') as f:
    np.save(f, feat_2d)

