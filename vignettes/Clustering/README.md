## Mouse cortex (Zeisel et al, 2018)
We demonstrate how OCAT sparsely encodes single-cell gene expression data using 3,005 cells and 4,412 genes in the mouse somatosensory cortex and hippocampal CA1 region from Zeisel et al. (2008). 

## Table of Contents
- [Step 0. Import data](#data_import)
- [Step 1. Data pre-processing](#pre_processing)
- [Step 2. Dimension reduction](#dim_reduct)
- [Step 3. Contruct bipartite graph through ghost cells](#ghost_cell)
- [Step 4. Clustering \& visualization](#clustering)
- [Step 5. Gene prioritization](#gene_prior)

<a name="data_import"></a>**Step 0. Import data**     
```python
data = loadmat('./Test_5_Zeisel.mat')
in_X = csr_matrix(data['in_X'])
gene_label = data['label2']
labels_combined = data['true_labs']
ds_combined = labels_combined.flatten()
```

<a name="pre_processing"></a>**Step 1. Data pre-processing**
```python
data_list = preprocess(data_list, log_norm=True, l2_norm=True)
```
<a name="dim_reduct"></a>**Step 2. Dimension reduction**

`dim` is the dimension of the subspace that the original gene expression vector is reduced to. 

```python
## dim = 50
data_list = apply_dim_reduct(data_list, dim = 50, mode='FSM', random_seed=42)
```

<a name="ghost_cell"></a>**Step 3. Contruct bipartite graph through ghost cells**
```python
## m = 80
ZW = sparse_encoding_integration_original(data_list, m = 80)
ZW_ = post_processing_pca(ZW)
```

<a name="clustering"></a>**Step 4. Clustering \& visualization**

```python
evaluate(ZW_, labels_combined, ds_combined, mode='ZW_', random_seed=42)
embedding = TSNE(n_components=2).fit_transform(W)
```
<img src="https://github.com/bowang-lab/OCAT/blob/master/vignettes/Clustering/Zeisel_clustering_v2.png" width="400" height="400" />  

<a name="clustering"></a>**Step 5. Gene prioritization**
```python
calculate_marker_gene(data, labels, topn=5, gene_labels, save_fig = None, save_csv = None)
```
