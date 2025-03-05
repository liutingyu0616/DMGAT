import numpy as np
import pandas as pd
import copy

adj = pd.read_csv('data/ncrna-drug_split.csv', index_col=0)
dataset = pd.read_excel('data/NoncoRNA_2020-02-10.xlsx')
dataset = dataset.dropna(subset=['drug_id'])
dataset = dataset.drop_duplicates(subset='drug_name')
drug_id = dataset['drug_id'].tolist()
drug_name = dataset['drug_name'].tolist()

adj_drug_name_list = adj.columns.tolist()
drug_name2id0 = {}
drug_name2id = {}
for i, name in enumerate(drug_name):
    drug_name2id0[name] = drug_id[i][:7]

for name in adj_drug_name_list:
    if name in drug_name:
        drug_name2id[name] = drug_name2id0[name]
    else:
        drug_name2id[name] = 'NotFound'
df = pd.DataFrame(list(drug_name2id.items()), columns=['name', 'id'])
df.to_csv('all_drug_id0.csv', index=False)

all_drug_id = pd.read_csv('all_drug_id.csv')
drug_name2id = dict(zip(all_drug_id['name'], all_drug_id['id']))

drug_name2id_value = list(drug_name2id.values())

import xml.etree.ElementTree as ET
tree = ET.parse('data/full database.xml')
root = tree.getroot()
ns = {'drugbank': 'http://www.drugbank.ca'}

def find_smiles(element, ns):
    if 'kind' in element.tag and element.text == 'SMILES':
        return element
    for child in element:
        result = find_smiles(child, ns)
        if result is not None:
            return result
    return None

all_drugs = root.findall(".//drugbank:drug", ns)
# Iterate through each drug entry
drugs = drug_name2id.copy()
drug_value = drugs.values()
for drug in all_drugs:
    # Find the primary drugbank-id
    primary = drug.find(".//drugbank:drugbank-id[@primary='true']", ns)
    if primary is not None:
        primary_id = primary.text

        if primary_id in drug_value:

            smiles_element = drug.find(".//drugbank:property[drugbank:kind='SMILES']/drugbank:value", ns)
            if smiles_element is not None:
                name = drug.find(".//drugbank:name", ns).text
                smiles = smiles_element.text
                # Add the results to the list
                for key, value in drugs.items():
                    if value == primary_id:
                        drugs[key] = smiles
                        break
            else:
                for key, value in drugs.items():
                    if value == primary_id:
                        drugs[key] = 'NotFound'
                        break

df = pd.DataFrame(list(drugs.items()), columns=['name', 'smiles'])
df.to_csv('all_drug_smiles.csv', index=False)



a = 1