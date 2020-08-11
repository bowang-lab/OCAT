## Mouse cortex example (Zeisel et al, 2015)
We demonstrate how OCAT sparsely encodes single-cell gene expression data using 3,005 cells and 4,412 genes in the mouse somatosensory cortex and hippocampal CA1 region from Zeisel et al. (2015). 

## Table of Contents
- [Step 0. Import data](#data_import)
- [Step 1. Data pre-processing](#pre_processing)
- [Step 2. Dimension reduction](#dim_reduct)
- [Step 3. Contruct bipartite graph through ghost cells](#ghost_cell)
- [Step 4. Clustering \& visualization](#clustering)
- [Step 5. Gene prioritization](#gene_prior)

```python
import OCAT
import numpy as np
```

<a name="data_import"></a>**Step 0. Import data**     
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
data_list = OCAT.apply_dim_reduct(data_list, dim = 50, mode='FSM', random_seed=42)
```

<a name="ghost_cell"></a>**Step 3. Contruct bipartite graph through ghost cells**

OCAT constructs a sparsified bipartite graph to embed the gene expression of each single cell. `m` is the number of ghost cells that each single cell connects to. 

```python
ZW = OCAT.sparse_encoding_integration(data_list, m = 80)
```

<a name="clustering"></a>**Step 4. Clustering \& visualization**

```python
## import the annotated labels for the mouse cortex data
labels_true = data['true_labs']

## evaluate the clustering performance of the predicted labels
num_cluster = len(np.unique(labels_true))
labels_pred = evaluate(ZW, n_cluster=n_cluster)
```
<img src="https://github.com/bowang-lab/OCAT/blob/master/vignettes/Clustering/Zeisel_clustering_v2.png" width="400" height="400" />  

<a name="gene_prior"></a>**Step 5. Gene prioritization**

```python
## import the gene labels of the mouse cortex scRNA-seq data
gene_label = data['label2']

calculate_marker_gene(data, labels, topn=5, gene_labels, save_fig = None, save_csv = None)
```
