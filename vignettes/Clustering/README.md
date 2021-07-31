## Mouse cortex example (Zeisel et al, 2015)
We demonstrate how OCAT sparsely encodes single-cell gene expression data using 3,005 cells and 4,412 genes in the mouse somatosensory cortex and hippocampal CA1 region from Zeisel et al. (2015). 

## Table of Contents
- [Step 0. Import data](#data_import)
- [Step 1. Run OCAT](#run_ocat)
- [Step 2. Clustering \& visualization](#clustering)
- [Step 3. Differential gene analysis](#gene_prior)

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

<a name="pre_processing"></a>**Step 1. Run OCAT**

The `run_OCAT` function automates 
1. pre-processing of the raw gene expression matrix through log-transformation and normalization (using l2-norm) 
2. reduces the dimension of the raw gene expression to `dim = 30` subspace
3. returns `ZW`, the OCAT sparse encoding of the integrated datasets with `m = 50` "ghost" cells

```python
ZW = OCAT.run_OCAT(data_list, m_list = [50], dim=30, p=0.3, log_norm=True, l2_norm=True)
```

<a name="clustering"></a>**Step 2. Clustering \& visualization**

```python
## import the annotated labels for the mouse cortex data
labels_true = data['true_labs'].flatten()

## predict clustering labels for the cells
labels_pred = OCAT.evaluate_clusters(ZW, num_cluster=len(np.unique(labels_true)))

## evaluate the clustering performance of the predicted labels
from sklearn.metrics.cluster import normalized_mutual_info_score
NMI_cell_type = normalized_mutual_info_score(labels_true, labels_pred)
```
<img src="https://github.com/bowang-lab/OCAT/blob/master/vignettes/Clustering/zeisel_github.png" width="500" height="500"/>  

<a name="gene_prior"></a>**Step 3. Gene prioritization**

```python
import matplotlib.pyplot as plt

## import the gene labels of the mouse cortex scRNA-seq data
gene_label = data['label2'].flatten()
gene_df, fig = OCAT.calculate_marker_gene(data=in_X.T, labels=labels_pred, topn=5, gene_labels=gene_label)

## save the top 5 most differential genes for each cell type group
gene_df.to_csv('marker_gene.csv')
## save the heatmap visualizing the most differential genes for each cell type group
plt.savefig('marker_gene.png')
```
<img src="https://github.com/bowang-lab/OCAT/blob/master/img/marker_gene_JAN31.png" width="500" height="500"/>
