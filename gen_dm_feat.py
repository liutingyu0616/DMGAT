import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from pydiffmap import diffusion_map as dm

with open('feat_2d.npy', 'rb') as f:
    feat_2d = np.load(f,allow_pickle=True).flat[0]

lnc_emb_2d = feat_2d['lnc_emb_2d']
mi_emb_2d = feat_2d['mi_emb_2d']
drug_emb_2d = feat_2d['drug_emb_2d']

# length_phi = 15   #length of swiss roll in angular direction
# length_Z = 15     #length of swiss roll in z direction
# sigma = 0.1       #noise strength
# m = 10000         #number of samples
#
# # create dataset
# phi = length_phi*np.random.rand(m)
# xi = np.random.rand(m)
# Z = length_Z*np.random.rand(m)
# X = 1./6*(phi + sigma*xi)*np.sin(phi)
# Y = 1./6*(phi + sigma*xi)*np.cos(phi)
#
# swiss_roll = np.array([X, Y, Z]).transpose()
#
# # check that we have the right shape
# print(swiss_roll.shape)
neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}

lnc_dmap0 = dm.DiffusionMap.from_sklearn(n_evecs=int(len(lnc_emb_2d)/3), epsilon='bgh', alpha=1.0, neighbor_params=neighbor_params)
lnc_dmap = lnc_dmap0.fit_transform(lnc_emb_2d)
mi_dmap0 = dm.DiffusionMap.from_sklearn(n_evecs=int(len(mi_emb_2d)/4), epsilon='bgh', alpha=1.0, neighbor_params=neighbor_params)
mi_dmap = mi_dmap0.fit_transform(mi_emb_2d)
drug_dmap0 = dm.DiffusionMap.from_sklearn(n_evecs=int(len(drug_emb_2d)/3), epsilon='bgh', alpha=1.0, neighbor_params=neighbor_params)
drug_dmap = drug_dmap0.fit_transform(drug_emb_2d)


lnc_dmap_mean = lnc_dmap.mean(axis=0)
mi_dmap_mean = mi_dmap.mean(axis=0)
drug_dmap_mean = drug_dmap.mean(axis=0)

rna_seq = pd.read_csv(r"rna_seq.csv")
drug_seq=pd.read_csv('drug_smiles.csv')

lnc_seq = rna_seq[rna_seq['type']=='lncRNA']
mi_seq = rna_seq[rna_seq['type']=='miRNA']

lnc_not_found_indices = np.where(lnc_seq['seq'] == 'NotFound')[0]
mi_not_found_indices = np.where(mi_seq['seq'] == 'NotFound')[0]
drug_not_found_indices = np.where(drug_seq['smiles'] == 'NotFound')[0]

for idx in lnc_not_found_indices:
    lnc_dmap = np.insert(lnc_dmap, idx, lnc_dmap_mean, axis=0)

for idx in mi_not_found_indices:
    mi_dmap = np.insert(mi_dmap, idx, mi_dmap_mean, axis=0)

for idx in drug_not_found_indices:
    drug_dmap = np.insert(drug_dmap, idx, drug_dmap_mean, axis=0)

feat_dm = {}
feat_dm['lnc_dmap'] = lnc_dmap
feat_dm['mi_dmap'] = mi_dmap
feat_dm['drug_dmap'] = drug_dmap


with open('feat_dm.npy', 'wb') as f:
    np.save(f, feat_dm)

# from pydiffmap.visualization import embedding_plot, data_plot
#
# embedding_plot(mydmap, scatter_kwargs = {'c': dmap[:,0], 'cmap': 'Spectral'})
# data_plot(mydmap, dim=3, scatter_kwargs = {'cmap': 'Spectral'})
