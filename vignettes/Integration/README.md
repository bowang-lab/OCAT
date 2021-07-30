## Integrating Five Human Pancreatic scRNA-seq datasets 

This vignette demonstrates how OCAT integrates multiple scRNA-seq datasets. The human pancreas data set consists of five data sources for human pancreatic cells (Baron et al., Segerstolpe et al., Muraro et al., Wang et al., Xin et al.). 
Tran et al. removed cells with ambiguous annotations, and the resulting batches contain a total of 14,767 cells with 15 different cell types. This data set captures batch effect across multiple sequencing technologies.

## Table of Contents
- [Step 0. Import data](#data_import)
- [Step 1. Run OCAT](#run_OCAT)
- [Step 2. Clustering \& visualization](#clustering)


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

<a name="pre_processing"></a>**Step 1. Run OCAT**
OCAT automates (i) pre-processing of the raw gene expression matrix through log-transformation and normalization (using l2-norm); (ii) reduces the dimension of the raw gene expression to `dim` subspace;

```python
ZW = OCAT.sparse_encoding_integration(data_list, m_list = [100, 100, 100, 100, 100])
```
<a name="clustering"></a>**Step 2. Clustering \& visualization**

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
<img src="https://github.com/bowang-lab/OCAT/blob/master/vignettes/Integration/Prancreas_UMAP_github.png" width="1000" height="400" />  
