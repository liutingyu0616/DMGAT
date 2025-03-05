import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

drug_seq_ori=pd.read_csv('drug_smiles.csv')
drug_seq = drug_seq_ori[drug_seq_ori['smiles'] != 'NotFound']

rna_seq = pd.read_csv(r"rna_seq.csv")
rna_seq = rna_seq.drop('id', axis=1)

lnc_seq_ori = rna_seq[rna_seq['type']=='lncRNA']
lnc_seq = lnc_seq_ori[lnc_seq_ori['seq'] != 'NotFound']
lnc_seq = lnc_seq.drop('type', axis=1)
mi_seq_ori = rna_seq[(rna_seq['type']=='miRNA')]
mi_seq = mi_seq_ori[mi_seq_ori['seq'] != 'NotFound']
mi_seq = mi_seq.drop('type', axis=1)

def gen_tfidf_feat(rna_seq):
    rna_seq_dict = dict(rna_seq.values)

    data = []
    for name, seq in rna_seq_dict.items():
        data.append([name, seq])

    kmers = 3
    p_kmers_seq, name_list = [
        [i[1][j : j + kmers] for j in range(len(i[1]) - kmers + 1)] for i in data
    ], [i[0] for i in data]

    spaced_lst = [" ".join(sublist) for sublist in p_kmers_seq]
    tv = CountVectorizer()
    tv_fit = tv.fit_transform(spaced_lst)
    tv.get_feature_names_out()

    return tv_fit.A

lnc_tfidf_feat = gen_tfidf_feat(lnc_seq)
mi_tfidf_feat = gen_tfidf_feat(mi_seq)
drug_tfidf_feat = gen_tfidf_feat(drug_seq)

lnc_tfidf_mean = lnc_tfidf_feat.mean(axis=0)
mi_tfidf_mean = mi_tfidf_feat.mean(axis=0)
drug_tfidf_mean = drug_tfidf_feat.mean(axis=0)


lnc_not_found_indices = np.where(lnc_seq_ori['seq'] == 'NotFound')[0]
mi_not_found_indices = np.where(mi_seq_ori['seq'] == 'NotFound')[0]
drug_not_found_indices = np.where(drug_seq_ori['smiles'] == 'NotFound')[0]

for idx in lnc_not_found_indices:
    lnc_tfidf_feat = np.insert(lnc_tfidf_feat, idx, lnc_tfidf_mean, axis=0)

for idx in mi_not_found_indices:
    mi_tfidf_feat = np.insert(mi_tfidf_feat, idx, mi_tfidf_mean, axis=0)

for idx in drug_not_found_indices:
    drug_tfidf_feat = np.insert(drug_tfidf_feat, idx, drug_tfidf_mean, axis=0)


feat_dm = {}
feat_dm['lnc_dmap'] = lnc_tfidf_feat
feat_dm['mi_dmap'] = mi_tfidf_feat
feat_dm['drug_dmap'] = drug_tfidf_feat


with open('feat_bow.npy', 'wb') as f:
    np.save(f, feat_dm)
