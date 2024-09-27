import pandas as pd
import numpy as np
import math
import pickle


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def Getgauss_RNA(adjacentmatrix, nm):
    """
    MiRNA Gaussian interaction profile kernels similarity
    """
    KM = np.zeros((nm, nm))

    gamaa = 1
    sumnormm = 0
    for i in range(nm):
        normm = np.linalg.norm(adjacentmatrix[i]) ** 2
        sumnormm = sumnormm + normm
    gamam = gamaa / (sumnormm / nm)

    for i in range(nm):
        for j in range(nm):
            KM[i, j] = math.exp(
                -gamam * (np.linalg.norm(adjacentmatrix[i] - adjacentmatrix[j]) ** 2)
            )
    return KM


def Getgauss_drug(adjacentmatrix, nd):
    """
    Disease Gaussian interaction profile kernels similarity
    """
    KD = np.zeros((nd, nd))
    gamaa = 1
    sumnormd = 0
    for i in range(nd):
        normd = np.linalg.norm(adjacentmatrix[:, i]) ** 2
        sumnormd = sumnormd + normd
    gamad = gamaa / (sumnormd / nd)

    for i in range(nd):
        for j in range(nd):
            KD[i, j] = math.exp(
                -(
                    gamad
                    * (np.linalg.norm(adjacentmatrix[:, i] - adjacentmatrix[:, j]) ** 2)
                )
            )
    return KD


seed_everything(42)
adj_df = pd.read_csv(r"../ncrna-drug_split.csv", index_col=0)
adj = adj_df.values
num_p, num_d = adj.shape

pos_ij = np.argwhere(adj == 1)
unlabelled_ij = np.argwhere(adj == 0)
np.random.shuffle(pos_ij)
np.random.shuffle(unlabelled_ij)

fold_cnt = 0

pos_train_ij_list = []
pos_test_ij_list = []
unlabelled_train_ij_list = []
unlabelled_test_ij_list = []
p_gip_list = []
d_gip_list = []

p_gip = Getgauss_RNA(adj, num_p)
d_gip = Getgauss_drug(adj, num_d)
# p_gip = np.ones((num_p, num_p))
# d_gip = np.ones((num_d, num_d))

p_gip_list.append(p_gip)
d_gip_list.append(d_gip)

fold_info = {
    "p_gip_list": p_gip_list,
    "d_gip_list": d_gip_list,
}
with open("fold_info.pickle", "wb") as f:
    pickle.dump(fold_info, f)

# with open(r"fold_info.pickle", "rb") as f:
#     fold_info = pickle.load(f)
