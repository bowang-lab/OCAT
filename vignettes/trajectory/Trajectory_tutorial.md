## Trajectory and pseudotime inference with HSMM example 
We demonstrate how OCAT infers trajecotry and pseudotime using the human skeletal muscle myoblast (HSMM) dataset from Tran and Bader (2020). 

## Table of Contents
- [Step 0. Import data](#data_import)
- [Step 1. Data pre-processing](#pre_processing)
- [Step 2. Dimension reduction](#dim_reduct)
- [Step 3. Contruct bipartite graph through ghost cells](#ghost_cell)
- [Step 4. Clustering \& visualization](#clustering)
- [Step 5. Trajectory and pseudotime inference](#trajectory)

```python
import OCAT
import numpy as np
```

<a name="data_import"></a>**Step 0. Import data**     

To download the compiled dataset:
```bash
$ wget https://data.wanglab.ml/OCAT/HSMM.zip
$ unzip HSMM.zip 
```

Inside the `HSMM` folder, the data and labels are organized as such:
```
HSMM
├── HSMM_label.txt
├── HSMM.txt
├── time_points.txt
```

```python
from scipy.io import loadmat
from scipy.sparse import csr_matrix

data = loadmat('./Test_5_Zeisel.mat')
in_X = csr_matrix(data['in_X'])
data_list = [in_X]
```

<a name="pre_processing"></a>**Step 1. Data pre-processing**

The gene expression data is first pre-processed through log-transformation and normalization (using l2-norm). 

```python
data_list = OCAT.preprocess(data_list, log_norm=True, l2_norm=True)
```
<a name="dim_reduct"></a>**Step 2. Dimension reduction**

`dim` is the dimension of the subspace that the original gene expression vector is reduced to. OCAT adopts a fast and efficient dimension reduction method `mode = 'FSM'`, but the commonly used princial component analysis (`mode= 'PCA'`) is also implemented. 

```python
data_list = OCAT.apply_dim_reduct(data_list, dim = 30, mode='FSM', random_seed=42)
```

<a name="ghost_cell"></a>**Step 3. Contruct bipartite graph through ghost cells**

OCAT constructs a sparsified bipartite graph to embed the gene expression of each single cell. `m` is the number of ghost cells that each single cell connects to. 

```python
ZW = OCAT.sparse_encoding_integration(data_list, m_list = [50])
```

<a name="clustering"></a>**Step 4. Clustering \& visualization**

```python
## import the annotated labels for the mouse cortex data
labels_true = data['true_labs'].flatten()

## predict clustering labels for the cells
labels_pred = OCAT.evaluate_clusters(ZW, num_cluster=len(np.unique(labels_true)))

## evaluate the clustering performance of the predicted labels
from sklearn.metrics.cluster import normalized_mutual_info_score
NMI_cell_type = normalized_mutual_info_score(labels_true, labels_pred)
```
<img src="https://github.com/bowang-lab/OCAT/blob/master/vignettes/Clustering/Zeisel_clustering_v2.png" width="500" height="500"/>  

<a name="gene_prior"></a>**Step 5. Gene prioritization**

```python
import matplotlib.pyplot as plt

## import the gene labels of the mouse cortex scRNA-seq data
gene_label = data['label2'].flatten()
gene_df, fig = OCAT.calculate_marker_gene(in_X.T, labels_pred, 5, gene_label, vmin=0, vmax=5)
gene_df.to_csv('marker_gene.csv')
plt.savefig('marker_gene.png')
```
<img src="https://github.com/bowang-lab/OCAT/blob/master/img/marker_gene_JAN31.png" width="500" height="500"/>
