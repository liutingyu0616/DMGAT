import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, lnc_emb, mi_emb, drug_emb, out_size):
        super().__init__()
        self.circ_emb = nn.Parameter(torch.randn(2, out_size))
        self.linear_lnc = nn.Linear(lnc_emb.shape[-1], out_size)
        self.linear_mi = nn.Linear(mi_emb.shape[-1], out_size)
        self.pi_emb = nn.Parameter(torch.randn(1, out_size))
        self.linear_drug = nn.Linear(drug_emb.shape[-1], out_size)
        # self.rnd_r = torch.rand(625, out_size).to('cuda')-0.5
        # self.rnd_d = torch.rand(121, out_size).to('cuda')-0.5

    def forward(self, lnc_emb, mi_emb, drug_emb):
        new_lnc_emb = self.linear_lnc(lnc_emb)
        new_mi_emb = self.linear_mi(mi_emb)
        drug_emb = self.linear_drug(drug_emb)
        rna_emb = torch.concat([self.circ_emb, new_lnc_emb, new_mi_emb, self.pi_emb], dim=0)
        return rna_emb, drug_emb

class GCN(nn.Module):
    """
        GCN layer

        Args:
            input_dim (int): Dimension of the input
            output_dim (int): Dimension of the output (a softmax distribution)
    """

    def __init__(self, in_dim, out_dim, adj):
        super(GCN, self).__init__()
        self.input_dim = in_dim
        self.output_dim = out_dim

        D_diag = adj.sum(1)

        # Create D^{-1/2}
        self.D_neg_sqrt = torch.diag_embed(torch.pow(D_diag, -0.5))

        # Initialise the weight matrix as a parameter
        self.W = nn.Parameter(torch.rand(in_dim, out_dim))

    def forward(self, feat, adj):
        # D^-1/2 * (A_hat * D^-1/2)
        support_1 = torch.matmul(self.D_neg_sqrt, torch.matmul(adj, self.D_neg_sqrt))

        # (D^-1/2 * A_hat * D^-1/2) * (X * W)
        support_2 = torch.matmul(support_1, torch.matmul(feat, self.W))

        # ReLU(D^-1/2 * A_hat * D^-1/2 * X * W)
        H = F.relu(support_2)

        return H

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.rnd = 1e5*(torch.rand(in_dim, out_features).to('cuda')-0.5)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -1e10 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, Wh)
        h_prime = (h_prime + self.rnd)/1e5

        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[: self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features :, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GAT(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        adj_full,
        dropout,
        alpha,
        nheads,
    ):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.bn_in = nn.BatchNorm1d(in_dim)
        assert hid_dim % nheads == 0
        nhid_per_head = int(hid_dim / nheads)
        self.layer1 = [
            GraphAttentionLayer(adj_full.shape[0], in_dim, nhid_per_head, dropout=dropout, alpha=alpha)
            for _ in range(nheads)
        ]
        for i, head in enumerate(self.layer1):
            self.add_module("layer1_head_{}".format(i), head)

        self.out_att = GraphAttentionLayer(
            adj_full.shape[0], hid_dim, out_dim, dropout=dropout, alpha=alpha
        )


    def forward(self, p_feat, d_feat, adj):
        x = torch.cat((p_feat, d_feat), dim=0)
        # x = self.bn_in(x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.layer1], dim=1)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)

        p_feat = x[: p_feat.shape[0], :]
        d_feat = x[p_feat.shape[0] :, :]
        return p_feat, d_feat





class Predictor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.rna_layer = nn.Linear(in_dim, hidden_dim)
        self.drug_layer = nn.Linear(in_dim, hidden_dim)

    def forward(self, p_feat, d_feat):
        new_p_feat = self.rna_layer(p_feat)
        new_d_feat = self.drug_layer(d_feat)
        res = new_p_feat.mm(new_d_feat.t())
        return F.sigmoid(res)


class PUTransGCN(nn.Module):
    def __init__(self, linear, r_gcn, d_gcn, gat, predictor, **kwargs):
    # def __init__(self, linear, gcn, p_encoder, d_encoder, predictor, **kwargs):
        super(PUTransGCN, self).__init__(**kwargs)
        self.linear = linear
        self.r_gcn = r_gcn
        self.d_gcn = d_gcn
        self.gat = gat
        self.predictor = predictor

    def forward(self, lnc_emb, mi_emb, drug_emb, rna_sim, drug_sim, adj_full):
        rna_feat_linear, drug_feat_linear = self.linear(lnc_emb, mi_emb, drug_emb)
        # p_feat_gcn = self.r_gcn(rna_feat_linear, rna_sim)
        # d_feat_gcn = self.d_gcn(drug_feat_linear, drug_sim)
        p_feat_gat, d_feat_gat = self.gat(rna_feat_linear, drug_feat_linear, adj_full)
        return self.predictor(p_feat_gat, d_feat_gat)