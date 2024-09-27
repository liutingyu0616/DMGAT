import numpy as np
import pandas as pd
from Bio import SeqIO
import re

adj = pd.read_csv('ncrna-drug_split.csv', index_col=0)
dataset = pd.read_excel('data/NoncoRNA_2020-02-10.xlsx')
ncrna_id = dataset['ncrna_id'].tolist()
ncrna_name = dataset['ncrna_name'].tolist()
ncrna_type = dataset['ncrna_type'].tolist()
drug_id = dataset['drug_id'].tolist()
drug_name = dataset['drug_name'].tolist()


miRNA_name2seq_dict = {}
with open('data/hairpin.fa') as handle:
    for record in SeqIO.parse(handle, "fasta"):
        miRNA_seq = str(record.seq)
        miRNA_name2seq_dict[record.name] = miRNA_seq

adj_rna_name_list = adj.index.tolist()

hsa_adj_rna_name_list = ['hsa-' + re.sub(r'-\d+p$', '', s) for s in adj_rna_name_list]
hsa_adj_rna_name_list = [re.sub('-miR-', '-mir-', s) for s in hsa_adj_rna_name_list]
hsa_adj_rna_name_list = [re.sub('1273g', '1273c', s) for s in hsa_adj_rna_name_list]
hsa_adj_rna_name_list = [re.sub('17-92', '17', s) for s in hsa_adj_rna_name_list]
hsa_adj_rna_name_list = [s.rstrip('*') for s in hsa_adj_rna_name_list]
miRNA_fa_names = list(miRNA_name2seq_dict.keys())
miRNA_names_inter = list(set(hsa_adj_rna_name_list).intersection(set(miRNA_fa_names)))

adj_rna_id_list = []
adj_rna_type_list = []
adj_rna_seq_list = []

for i, name in enumerate(adj_rna_name_list):
    if name in ncrna_name:
        index = ncrna_name.index(name)
        adj_rna_id_list.append(ncrna_id[index])
        adj_rna_type_list.append(ncrna_type[index])
    else:
        adj_rna_id_list.append('NotFound')
        adj_rna_type_list.append('NotFound')
        # adj_rna_seq_list.append('NotFound')
    hsa_adj_rna_name = hsa_adj_rna_name_list[i].strip()
    if hsa_adj_rna_name in miRNA_fa_names:
        adj_rna_seq_list.append(miRNA_name2seq_dict[hsa_adj_rna_name])
    elif hsa_adj_rna_name+'-1' in miRNA_fa_names:
        adj_rna_seq_list.append(miRNA_name2seq_dict[hsa_adj_rna_name+'-1'])
    elif hsa_adj_rna_name+'a' in miRNA_fa_names:
        adj_rna_seq_list.append(miRNA_name2seq_dict[hsa_adj_rna_name+'a'])
    elif hsa_adj_rna_name+'b-1' in miRNA_fa_names:
        adj_rna_seq_list.append(miRNA_name2seq_dict[hsa_adj_rna_name+'b-1'])
    elif hsa_adj_rna_name+'a-1' in miRNA_fa_names:
        adj_rna_seq_list.append(miRNA_name2seq_dict[hsa_adj_rna_name+'a-1'])

    else:
        adj_rna_seq_list.append('NotFound')

result = list(zip(adj_rna_name_list, adj_rna_id_list, adj_rna_type_list, adj_rna_seq_list))
result = np.array(result)

column_names = ['name', 'id', 'type', 'seq']

# 创建DataFrame
df = pd.DataFrame(result, columns=column_names)

# 保存为CSV文件
df.to_csv('rna_seq0.csv', index=False)
