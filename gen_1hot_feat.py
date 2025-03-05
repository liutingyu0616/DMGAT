import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from pydiffmap import diffusion_map as dm

adj_df = pd.read_csv(r"ncrna-drug_split.csv", index_col=0)
adj = adj_df.values

rna_seq = pd.read_csv(r"rna_seq.csv")
drug_seq=pd.read_csv('drug_smiles.csv')

lnc_seq = rna_seq[rna_seq['type']=='lncRNA']
mi_seq = rna_seq[rna_seq['type']=='miRNA']

n_lnc = len(lnc_seq)
n_mi = len(mi_seq)

lnc_1hot = adj[:n_lnc, :]
mi_1hot = adj[n_lnc:, :]
drug_1hot = adj.T

feat_dmap = {}
feat_dmap['lnc_dmap'] = lnc_1hot
feat_dmap['mi_dmap'] = mi_1hot
feat_dmap['drug_dmap'] = drug_1hot

with open('feat_1hot.npy', 'wb') as f:
    np.save(f, feat_dmap)
