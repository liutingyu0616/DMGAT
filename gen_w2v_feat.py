import numpy as np
import pandas as pd
from gensim.models import Word2Vec



def gen_w2v_feat(rna_seq, vector_size, type):
    rna_seq_dict = dict(rna_seq.values)

    data = []
    for name, seq in rna_seq_dict.items():
        data.append([name, seq])

    kmers = 3
    p_kmers_seq, name_list = [
        [i[1][j : j + kmers] for j in range(len(i[1]) - kmers + 1)] for i in data
    ], [i[0] for i in data]
    name2id, id2name = {}, []

    cnt = 0
    for name in name_list:
        if name not in name2id:
            name2id[name] = cnt
            id2name.append(name)
            cnt += 1
    num_class = cnt
    kmers2id, id2kmers = {"<EOS>": 0}, ["<EOS>"]

    kmers_cnt = 1
    for kmers_seq in p_kmers_seq:
        for kmers in kmers_seq:
            if kmers not in kmers2id:
                kmers2id[kmers] = kmers_cnt
                id2kmers.append(kmers)
                kmers_cnt += 1
    num_kmers = kmers_cnt

    name_id_list = np.array([name2id[i] for i in name_list], dtype="int32")
    p_seq_len = np.array([len(s) + 1 for s in p_kmers_seq], dtype="int32")
    max_seq_len = p_seq_len.max()
    tokenized_seq = np.array([[kmers2id[i] for i in s] for s in p_kmers_seq], dtype=object)
    pad_kmers_id_seq = np.zeros((tokenized_seq.shape[0], max_seq_len), dtype=int)
    for i, seq in enumerate(tokenized_seq):
        pad_seq = np.pad(seq, (0, max_seq_len - len(seq)), constant_values=0)
        pad_kmers_id_seq[i] = pad_seq

    vector = {}

    doc = [i + ["<EOS>"] for i in p_kmers_seq]

    window = 10
    workers = 4
    model = Word2Vec(
        doc,
        min_count=0,
        window=window,
        vector_size=vector_size,
        workers=workers,
        sg=1,
        epochs=500,
    )
    p_kmers_emb = np.zeros((num_kmers, vector_size), dtype=np.float32)
    for i in range(num_kmers):
        p_kmers_emb[i] = model.wv[id2kmers[i]]

    # set(p_in_adj) <= set(name_list)
    # intersection = list(set(p_in_adj) & set(name_list))
    # list(set(p_in_adj).difference(set(intersection)))
    gensim_feat = {"kmers_emb": p_kmers_emb, "pad_kmers_id_seq": pad_kmers_id_seq}
    np.save(f"gensim_feat_{type}_{vector_size}.npy", gensim_feat)
    # np.save("gensim_pad_tokenized_seq.npy", pad_tokenized_seq)

drug_seq=pd.read_csv('drug_smiles.csv')
drug_seq = drug_seq[drug_seq['smiles'] != 'NotFound']
gen_w2v_feat(drug_seq, 128, 'drug')

rna_seq = pd.read_csv(r"rna_seq.csv")
rna_seq = rna_seq.drop('id', axis=1)

lnc_seq = rna_seq[rna_seq['type']=='lncRNA']
lnc_seq = lnc_seq[lnc_seq['seq'] != 'NotFound']
lnc_seq = lnc_seq.drop('type', axis=1)
mi_seq = rna_seq[(rna_seq['type']=='miRNA')]
mi_seq = mi_seq[mi_seq['seq'] != 'NotFound']
mi_seq = mi_seq.drop('type', axis=1)

gen_w2v_feat(lnc_seq, 128, 'lnc')
gen_w2v_feat(mi_seq, 128, 'mi')
