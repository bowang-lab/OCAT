## Trajectory and pseudotime inference with the HSMM dataset
We demonstrate how OCAT infers trajectory and pseudotime using the human skeletal muscle myoblast (HSMM) dataset from Tran and Bader (2020). 

## Table of Contents
- [Step 0. Import data](#data_import)
- [Step 1. Data pre-processing](#pre_processing)
- [Step 2. Dimension reduction](#dim_reduct)
- [Step 3. Contruct bipartite graph through ghost cells](#ghost_cell)
- [Step 4. Clustering](#clustering)
- [Step 5. Trajectory inference](#trajectory)
- [Step 6. Pseudotime inference](#pseudo)


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
```

```python
import pandas as pd

data = pd.read_csv('./HSMM/HSMM.txt', delimiter=' ')

#Transpose data matrix to cell by gene
data = data.T
data = csr_matrix(data)
data_list = [data]
```

<a name="pre_processing"></a>**Step 1. Data pre-processing**

The gene expression data is first pre-processed through log-transformation and normalization (using l2-norm). 

```python
data_list = OCAT.preprocess(data_list, log_norm=True, l2_norm=True)
```
<a name="dim_reduct"></a>**Step 2. Dimension reduction**

`dim` is the dimension of the subspace that the original gene expression vector is reduced to. OCAT adopts a fast and efficient dimension reduction method `mode = 'FSM'`, but the commonly used princial component analysis (`mode= 'PCA'`) is also implemented. 

```python
data_list = OCAT.apply_dim_reduct(data_list, dim=100, mode='FSM', random_seed=42, upsample=False)
```

<a name="ghost_cell"></a>**Step 3. Contruct bipartite graph through ghost cells**

OCAT constructs a sparsified bipartite graph to embed the gene expression of each single cell. `m` is the number of ghost cells that each single cell connects to. 

```python
ZW = OCAT.sparse_encoding_integration(data_list, m_list=[25])
```

<a name="clustering"></a>**Step 4. Clustering**

```python
label = pd.read_csv('./HSMM/HSMM_label.txt', delimiter=' ')
labels_combined_c = np.array(label.loc[:,'V1'])
mapping = {1: 'Fibroblast', 2:'Myotubes', 3: 'Myoblasts', 4:'Undiff', 5:'Intermediates'}
labels_combined = [mapping[i] for i in labels_combined_c]
labels_combined = np.array(labels_combined)

num_cluster = len(np.unique(labels_combined))
pca = KMeans(n_clusters=num_cluster, n_init=20).fit(ZW)
nmi = normalized_mutual_info_score(labels_combined, pca.labels_)
embedding = save_coordinates(ZW, save_path='./', save_name='Z_coordinates_X.txt', labels_combined=labels_combined)
```


<a name="trajectory"></a>**Step 5. Trajectory inference**

`OCAT.compute_lineage()` function infers `Lineages` over clusters with the OCAT features, predicted/true cluster labels and a user-specified `root_cluster`.

```python
Lineage, root_cluster, cluster_labels, tree = OCAT.compute_lineage(ZW, labels_combined, root_cluster='Myoblasts', name='OE', reverse=0)
```
<img src="https://github.com/bowang-lab/OCAT/blob/master/img/trajectory.png" width="350" height="350"/>

<a name="pseudo"></a>**Step 6. Pseudotime inference**

`OCAT.compute_ptime()` function infers pseudotime for individual cell using the OCAT extracted features and the predicted lineage. `OCAT.draw_Ptime()` function visualizes the inferred lineages and pseudotime.

```python
Ptime, root_cell_list = OCAT.compute_ptime(ZW, labels_combined, Lineage, root_cluster, embedding)

draw_Ptime(Ptime, Lineage, root_cell_list, labels_combined, labels_combined_c, embedding, './ptime.png', 'ptime.png')
```
<img src="https://github.com/bowang-lab/OCAT/blob/master/img/ptime.png" width="500" height="500"/>
