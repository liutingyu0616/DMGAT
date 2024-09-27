import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


with open('feat_dm.npy', 'rb') as f:
    feat_dm = np.load(f,allow_pickle=True).flat[0]

lnc_dmap = feat_dm['lnc_dmap']
mi_dmap = feat_dm['mi_dmap']
drug_dmap = feat_dm['drug_dmap']

lnc_dmap_distances = pdist(lnc_dmap, metric='euclidean')
mi_dmap_distances = pdist(mi_dmap, metric='euclidean')
drug_dmap_distances = pdist(drug_dmap, metric='euclidean')

# 将距离结果转换为方阵形式，便于查看
lnc_distance_matrix = squareform(lnc_dmap_distances)
mi_distance_matrix = squareform(mi_dmap_distances)
drug_distance_matrix = squareform(drug_dmap_distances)

circ_sim = np.array([[1, 0], [0, 1]])
lnc_sim = np.e**(-lnc_distance_matrix)/max((np.e**(-lnc_distance_matrix)).sum(1))
np.fill_diagonal(lnc_sim, 1)
mi_sim = np.e**(-mi_distance_matrix)/max((np.e**(-mi_distance_matrix)).sum(1))
np.fill_diagonal(mi_sim, 1)
pi_sim = np.array([[1]])

rna_self_sim = np.block([
    [circ_sim, np.zeros((circ_sim.shape[0], lnc_sim.shape[1])), np.zeros((circ_sim.shape[0], mi_sim.shape[1])), np.zeros((circ_sim.shape[0], pi_sim.shape[1]))],
    [np.zeros((lnc_sim.shape[0], circ_sim.shape[1])), lnc_sim, np.zeros((lnc_sim.shape[0], mi_sim.shape[1])), np.zeros((lnc_sim.shape[0], pi_sim.shape[1]))],
    [np.zeros((mi_sim.shape[0], circ_sim.shape[1])), np.zeros((mi_sim.shape[0], lnc_sim.shape[1])), mi_sim, np.zeros((mi_sim.shape[0], pi_sim.shape[1]))],
    [np.zeros((pi_sim.shape[0], circ_sim.shape[1])), np.zeros((pi_sim.shape[0], lnc_sim.shape[1])), np.zeros((pi_sim.shape[0], mi_sim.shape[1])), pi_sim]
])

drug_self_sim = np.e**(-drug_distance_matrix)/max((np.e**(-drug_distance_matrix)).sum(1))
np.fill_diagonal(drug_self_sim, 1)

self_sim = {}
self_sim['rna_self_sim'] = rna_self_sim
self_sim['drug_self_sim'] = drug_self_sim

with open('self_sim.npy', 'wb') as f:
    np.save(f, self_sim)



a=1