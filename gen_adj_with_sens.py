import pandas as pd
import numpy as np
import copy

DR_Curated = pd.read_excel('ver1/data/DR_Curated.xlsx')
all_drug_id = pd.read_csv('temp_data/all_drug_id.csv')
rna_seq = pd.read_csv('rna_seq.csv')

rna_name_list = rna_seq['name'].tolist()
rna_id_list = rna_seq['id'].tolist()
drug_name_list = all_drug_id['name'].tolist()
drug_id_list = all_drug_id['id'].tolist()

rna_name2id = dict(zip(rna_name_list, rna_id_list))
drug_name2id = dict(zip(drug_name_list, drug_id_list))

columns_to_keep = ['ncRNA_Name', 'ENSEMBL_ID', 'miRBase_ID', 'ncRNA_Type', 'Drug_Name', 'DrugBank_ID', 'Effect']
DR_Curated = DR_Curated[columns_to_keep]

full_adj = pd.DataFrame(0, index=rna_name_list, columns=drug_name_list)
for index in full_adj.index:
    for column in full_adj.columns:
        rna_id = rna_name2id[index]
        drug_id = drug_name2id[column]
        if rna_id != 'NotFound' and drug_id != 'NotFound':
            rna_drug_ass = DR_Curated[((DR_Curated['ENSEMBL_ID'] == rna_id) |
                       (DR_Curated['miRBase_ID'] == rna_id)) &
                       (DR_Curated['DrugBank_ID'] == drug_id)]
            if len(rna_drug_ass)>0:
                n_sensitive = (rna_drug_ass['Effect'] == 'sensitive').sum()
                n_resistant = (rna_drug_ass['Effect'] == 'resistant').sum()
                if n_sensitive>n_resistant:
                    full_adj.at[index, column] = -1
                else:
                    full_adj.at[index, column] = 1

adj = pd.read_csv('ncrna-drug_split.csv', index_col=0)
adj_copy = adj.copy()

adj_copy[(adj == 0) & (full_adj == -1)] = -1
adj_copy.to_csv('adj_with_sens.csv')

'''
(adj==1).sum().sum()
Out[26]: 2693
121*625
Out[27]: 75625
2693/75625
Out[28]: 0.0356099173553719
((adj==1)&(full_adj==-1)).sum().sum()
Out[31]: 81
((adj==1)&(full_adj==1)).sum().sum()
Out[32]: 198
((adj==0)&(full_adj==1)).sum().sum()
Out[33]: 561
((adj==0)&(full_adj==-1)).sum().sum()
Out[34]: 408
((adj==1)&(full_adj==-1)).sum().sum()
Out[35]: 81
(full_adj==-1).sum().sum()
Out[36]: 489
(full_adj==1).sum().sum()
Out[37]: 759
'''
