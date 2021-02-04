## Integrating Five Human Pancreatic scRNA-seq datasets 

This vignette demonstrates how OCAT integrates multiple scRNA-seq datasets. The human pancreas data set consists of five data sources for human pancreatic cells (Baron et al., Segerstolpe et al., Muraro et al., Wang et al., Xin et al.). 
Tran et al. removed cells with ambiguous annotations, and the resulting batches contain a total of 14,767 cells with 15 different cell types. This data set captures batch effect across multiple sequencing technologies.

## Table of Contents
- [Step 0. Import data](#data_import)
- [Step 1. Data pre-processing](#pre_processing)
- [Step 2. Dimension reduction](#dim_reduct)
- [Step 3. Contruct bipartite graph through ghost cells](#ghost_cell)
- [Step 4. Clustering \& visualization](#clustering)


<a name="data_import"></a>**Step 0. Import data**   
The Human Pancreas dataset consists of five scRNA-seq datasets (Baron et al. 2016, Muraro et al. 2016, Segerstolpe et al. 2016, Wang et al. 2016, Xin et al. 2016). 

To download the compiled dataset:
```bash
$ wget https://data.wanglab.ml/OCAT/Pancreas.zip
$ unzip Pancreas.zip 
```

Inside the `Pancreas` folder, the data and labels are organized as such:
```
Pancreas
├── data
│   ├── baron_1.npz
│   ├── muraro_2.npz
│   ├── seg_3.npz
│   ├── wang_4.npz
│   └── xin_5.npz
└── label
    ├── baron_1_label.npy
    ├── muraro_2_label.npy
    ├── seg_3_label.npy
    ├── wang_4_label.npy
    └── xin_5_label.npy
```
    
```python
import os
from scipy.sparse import load_npz, csr_matrix

data_path = './Pancreas/data'
file_list = ['baron_1', 'muraro_2', 'seg_3', 'wang_4', 'xin_5']
data_list = [load_npz(os.path.join(data_path, i + '.npz')).tocsr() for i in file_list]
```

```python
import OCAT
import numpy as np
```

<a name="pre_processing"></a>**Step 1. Data pre-processing**

The gene expression data is first pre-processed through log-transformation and normalization (using l2-norm). 

```python
data_list = OCAT.preprocess(data_list, log_norm=True, l2_norm=True)
```
<a name="dim_reduct"></a>**Step 2. Dimension reduction**

`dim` is the dimension of the subspace that the original gene expression vector is reduced to. OCAT adopts a fast and efficient dimension reduction method (`mode = 'FSM'`), while the commonly used princial component analysis (`mode= 'PCA'`) is also implemented. 

```python
data_list = OCAT.apply_dim_reduct(data_list, dim = 50, mode='FSM', random_seed=42)
```

<a name="ghost_cell"></a>**Step 3. Contruct bipartite graph through ghost cells**

OCAT constructs a sparsified bipartite graph to embed the gene expression of each single cell. `m_list` is the number of ghost cells that each single cell connects to. 

```python
ZW = OCAT.sparse_encoding_integration(data_list, m_list = [100, 100, 100, 100, 100])
```

<a name="clustering"></a>**Step 4. Clustering \& visualization**

```python
## import the annotated labels for the pancreas data
label_path = './Pancreas/label'
label_list = [np.load(os.path.join(label_path, i + '_label.npy'), allow_pickle=True) for i in file_list]
labels_combined = np.concatenate(label_list, axis=0)

## create dataset labels for the pancreas data
def create_ds_label(label_list, file_list):
    label_ds_list = []
    for i, name in enumerate(file_list):
        label_ds_temp = np.repeat(name, len(label_list[i]))
        label_ds_list.append(label_ds_temp)
    return label_ds_list

label_ds_list = create_ds_label(label_list, file_list)
ds_combined = np.concatenate(label_ds_list, axis=0)

## evaluate the clustering performance of the predicted labels
num_cluster = len(np.unique(labels_combined))
labels_pred = evaluate(ZW, n_cluster=n_cluster)
```
<img src="https://github.com/bowang-lab/OCAT/blob/master/vignettes/Integration/pancreas_integration.png" width="800" height="400" />  
