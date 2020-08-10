## Mouse cortex (Zeisel et al, 2018)

## Table of Contents
- [Step 0. Import data](#data_import)
- [Step 1. Data pre-processing](#pre_processing)
- [Step 2. Dimension reduction](#dim_reduct)
- [Step 3. Contruct bipartite graph through ghost cells](#ghost_cell)
- [Step 4. Clustering \& visualization](#clustering)

<a name="data_import"></a>**Step 0. Import data**     
```python
data = loadmat('./Test_5_Zeisel.mat')
in_X = csr_matrix(data['in_X'])
gene_label = data['label2']
labels_combined = data['true_labs']
```

<a name="pre_processing"></a>**Step 1. Data pre-processing**
```python
data_list = preprocess(data_list, log_norm=True, l2_norm=True)
```
<a name="dim_reduct"></a>**Step 2. Dimension reduction**
```python
data_list = apply_dim_reduct(data_list, dim=dim, mode='FSM', random_seed=42, upsample=False)
```

<a name="ghost_cell"></a>**Step 3. Contruct bipartite graph through ghost cells**
```python
ZW = sparse_encoding_integration_original(data_list, m)
ZW_ = post_processing_pca(ZW)
```

<a name="clustering"></a>**Step 4. Clustering \& visualization**

```python
evaluate(ZW_, labels_combined, ds_combined, mode='ZW_', random_seed=random_seed)
embedding = TSNE(n_components=2).fit_transform(W)
```
<img src="https://github.com/bowang-lab/OCAT/vignettes/Clustering/Zeisel_clustering_v2.png" width="400" height="400" />  
