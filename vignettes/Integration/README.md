## Integrating Six Human Pancreatic scRNA-seq datasets 

This vignette demonstrates how OCAT integrates multiple scRNA-seq datasets. The human pancreas data set consists of five data sources for human pancreatic cells with six datasets(Baron et al., Segerstolpe et al., Muraro et al., Wang et al., Xin et al.). 
Tran et al. removed cells with ambiguous annotations for all datasets except Muraro Celseq2, and the resulting batches contain a total of 14,767 cells with 15 different cell types. This data set captures batch effect across multiple sequencing technologies.

## Table of Contents
- [Step 0. Import data](#data_import)
- [Step 1. Run OCAT](#run_OCAT)
- [Step 2. Clustering \& visualization](#clustering)


<a name="data_import"></a>**Step 0. Import data**   
The Human Pancreas dataset consists of six scRNA-seq datasets (Baron et al. 2016, Muraro et al. 2016 (two from here), Segerstolpe et al. 2016, Wang et al. 2016, Xin et al. 2016). 

The compiled AnnData dataset can be downloaded [here](https://drive.google.com/file/d/1shc4OYIbq2FwbyGUaYuzizuvzW-giSTs/view).
    
```python
import OCAT
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt

data_path = 'fivepancreas_wang_raw.h5ad'
data = ad.read_h5ad(data_path)

ref_list = [data[data.obs.dataset == i,] for i in np.unique(data.obs.dataset)]
data_list = [i.X for i in ref_list]
```

<a name="pre_processing"></a>**Step 1. Run OCAT**

The `run_OCAT` function automates 
1. pre-processing of the raw gene expression matrix through log-transformation and normalization (using l2-norm) 
2. reduces the dimension of the raw gene expression to `dim = 60` subspace
3. returns `ZW`, the OCAT sparse encoding of the integrated datasets with `m = 65` "ghost" cells in each dataset

```python
m_list = [65]*5
ZW = OCAT.run_OCAT(ref_data_list, dim=60, m_list=m_list)
```
<a name="clustering"></a>**Step 2. Clustering \& visualization**

Retrieve the annotated labels and create batch labels for the pancreas data
```python
query_latent = ad.AnnData(ZW)
query_latent.obs['cell_type']=np.concatenate([i.obs.cell_type.tolist() for i in ref_list],axis=0)
query_latent.obs['dataset']=np.concatenate([i.obs.dataset.tolist() for i in ref_list],axis=0)
```

Evaluate the clustering performance of the predicted labels
```python
from sklearn.metrics.cluster import normalized_mutual_info_score
labels_pred = OCAT.evaluate_clusters(ZW, num_cluster=len(np.unique(query_latent.obs['cell_type'])))
batch_pred = OCAT.evaluate_clusters(ZW, num_cluster=len(np.unique(query_latent.obs['dataset'])))

NMI_cell_type = normalized_mutual_info_score(query_latent.obs['cell_type'], labels_pred)
NMI_batch = normalized_mutual_info_score(query_latent.obs['dataset'], batch_pred)
```
UMAP Visualization
```python
sc.pp.neighbors(query_latent)
sc.tl.leiden(query_latent)
sc.tl.umap(query_latent)
plt.figure()
sc.pl.umap(
    query_latent,
    color=["dataset", "cell_type"],
    frameon=False,
    wspace=0.6,
)
```
<img src="https://github.com/bowang-lab/OCAT/blob/master/vignettes/Integration/Pancreas_UMAP_github.png" width="1000" height="300" />  
