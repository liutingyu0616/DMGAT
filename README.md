# DMGAT


This is a PyTorch(an open-source deep learning framework) implementation with a model for ncRNA-drug resistance association prediction as described in the paper：

"DMGAT: Predicting ncRNA-Drug resistance associations based on diffusion map and heterogeneous graph attention network"

![Alt text](fig/ncRNA_flowchart.png?raw=true "DMGAT pipeline")

The flowchart of DMGAT. (a) ncRNA and drug node embedding and similarity calculation. Split sequence of ncRNA and SMILES of drug into several 3-mers, and then replaced by their corresponding feature vectors derived from word2vec. Flatten the feature to 2D using a deep learning network. Using diffusion map to reduce the dimension of feature. Sequence similarity obtained by calculating the Euclid distance of their feature. (b) Feature extraction using GCN. Contracting two separate homogeneous graph convolution networks to extracting ncRNA and drug feature respectively based on their diffusion map feature and similarity. (c) Predicting resistance association score using graph attention network. Combining feature of ncRNA and drug based on attention mechanism, multiply two feature matrices to get the predicted adjacency matrix.

# Getting Started

## Installation
Setup conda environment:
```
conda create -n DMGAT python=3.10 -y
conda activate DMGAT
```

Install required packages
```
pip install numpy==1.25.0
pip install scipy==1.11.1
pip install pandas==1.5.3
pip install openpyxl==3.0.10
pip install scikit-learn==1.2.2
pip install gensim==4.3.1
pip install tqdm==4.65.0
pip install jupyterlab==3.6.3
pip install matplotlib==3.8.2
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```


# Training

To train models for cross-validation, please follow the following steps:
1. Run `main.py` 
2. For the ablation experiments mentioned in the paper, run `main_noDM.py`, `main_noGCN.py` and `main_noGAT.py`
