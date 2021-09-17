# MolR

This repository is the PyTorch implementation of MolR ([arXiv]()):
> Chemical-reaction-aware Molecule Representation Learning  
Hongwei Wang, Weijiang Li, Xiaomeng Jin, Heng Ji, Kyunghyun Cho, Martin Burke, Jiawei Han  
arXiv Preprint

MolR uses graph neural networks (GNNs) as the molecule encoder, and preserves the equivalence of molecules w.r.t. chemical reactions in the embedding space.
Specifically, MolR forces the sum of the reactant embeddings and the sum of the product embeddings to be equal for each chemical reaction, which is shown to keep the embedding space well-organized and improve the generalization ability of the model.
MolR achieves substantial gains over state-of-the-art baselines.
Below is the result of Hit@1 on USPTO-479k and real reaction dataset for the task of chemical reaction prediction:

| Dataset      | USPTO-479k | real reaction |
| :---------: | :---: | :------:  |
| Mol2vec      | 0.614  | 0.313   |
| MolBERT      | 0.623  | 0.313   |
| __MolR-TAG__ | __0.882__ | __0.625__ |


Below is the result of AUC on BBBP, HIV, and BACE datasets for the task of molecule property prediction:

| Dataset      | BBBP  | HIV | BACE |
| :----------: | :---: | :---:  | :---: |
| Mol2vec      | 0.872  | 0.769  | 0.862 |
| MolBERT      | 0.762  | 0.783  | 0.866 |
| __MolR-GCN__ | __0.890__ | __0.802__ | __0.882__ |


Below is the result of RMSE on QM9 dataset for the task of graph-edit-distance prediction:

| Dataset      | QM9  |
| :----------: | :---: |
| Mol2vec      | 0.995  |
| MolBERT      | 0.937  |
| __MolR-SAGE__ | __0.817__ |

Below are the visualized reactions of alcohol oxidation and aldehyde oxidation using PCA:
<img src="https://github.com/hwwang55/MolR/blob/master/reaction.png" alt="drawing" width="400"/>

Below is the visualized molecule embedding space on BBBP dataset using t-SNE:
<img src="https://github.com/hwwang55/MolR/blob/master/space.png" alt="drawing" width="700"/>


For more results, please refer to our paper.

### Files in the folder

- `data/`
  - `USPTO-479k/` for chemical reaction prediction
    - `USPTO-479k.zip`: zipped file containing `train.csv`, `valid.csv`, and `test.csv`. Please unzip this file and put the three csv files under this directory. The cached files of USPTO-479k in DGL format is too large to be uploaded to GitHub. If you want to save the time of pre-processing this dataset, please download them [here](https://drive.google.com/file/d/1BcBlXOELDBUTticzsZTDHQthYP2ASJ5g/view?usp=sharing) and put the unzipped `/cache` directory under this directory.
  - `real_reaction_test/`  for chemical reaction prediction
    - `real_reaction_test.csv`: SMILES of reactants and multiple choices of 16 real questions
  - `BBBP/` for molecule property prediction and visualization
    - `BBBP.csv`: the original dataset file
    - `BBBP.bin`: the cached dataset in DGL format
    - `ged_wrt_1196.pkl`: the cached graph-edit-distance of all molecules w.r.t. No. 1196 molecule
    - `sssr.pkl`: the cached numbers of smallest rings for all molecules
  - `HIV/` for molecule property prediction
    - `HIV.csv`: the original dataset file. Note that the `HIV.bin` is too large to be uploaded to GitHub. If you want to save the time of pre-processing this dataset, please download it [here](https://drive.google.com/file/d/1xFE4BDyQtOWkABs3ufa7uetxz0MiFmh9/view?usp=sharing) and put the unzipped `HIV.bin` under this directory.
  - `BACE/` for molecule property prediction
    - `BACE.csv`: the original dataset
    - `BACE.bin`: the cached dataset in DGL format
  - `Tox21/` for molecule property prediction
    - `Tox21.csv`: the original dataset
    - `Tox21.bin`: the cached dataset in DGL format
  - `ClinTox/` for molecule property prediction
    - `ClinTox.csv`: the original dataset
    - `ClinTox.bin` the cached dataset in DGL format
  - `QM9/`  for graph-edit-distance prediction
    - `QM9.csv`: the original dataset
    - `pairwise_ged.csv`: the randomly selected 10,000 molecule pairs from the first 1,000 molecules in QM9.csv
    - `ged0.bin` and `ged1.bin`: cached molecule pairs in DGL format
- `src/`
  - `ged_pred/`: code for graph-edit-distance prediction task
  - `property_pred/`: code for molecule property prediction task
  - `real_reaction_test/`: code for real reaction test task
  - `visualization/`: code for embedding visualization task
  - `data_processing.py`: processing USPTO-479k dataset
  - `featurizer.py`: an easy-to-use API for converting a molecule SMILES to embedding
  - `main.py`: main function
  - `model.py`: implementation of GNNs
  - `train.py`: training procedure on USPTO-479k dataset
- `saved/` (pretrained models with the name format of `gnn_dim`)
  - `gat_1024/`
  - `gcn_1024/`
  - `sage_1024/`
  - `tag_1024/`


### Running the code

- If you would like to train the model from scratch or evaluate the pretrained model on downstream tasks of molecule property prediction, GED prediction, or visualization, just uncomment the corresponding part of code in `main.py` and run
  ```
  $ python main.py
  ```
  The default task is set as `--task=pretrain`. Hyper-parameter settings for other tasks are provided in  `main.py`, which should be easy to understand.  
  __Note__: If you spot CUDA_OUT_OF_MEMORY error under the default hyperparameter setting, please consider decreasing the batch size first (e.g., to 2048). This will decrease the performance but not too much.
- If you would like to use the pretrained model to process your own molecules and obtain their embeddings, please run
  ```
  $ python featurizer.py
  ```
  Please see `example_usage()` function in `featurizer.py` for details, which should be easy to understand.

### Required packages

The code has been tested running under Python 3.7 and CUDA 11.0, with the following packages installed (along with their dependencies):

- torch == 1.8.1
- dgl-cu110 == 0.6.1 (can also use dgl == 0.6.1 if run on CPU only)
- pysmiles == 1.0.1
- scikit-learn==0.24.2
- networkx==2.5.1
- matplotlib==3.4.2 (for `--task=visualization`)
- openbabel==3.1.1 (for `--task=visualization` and `--subtask=ring`)
- scipy==1.7.0
