## Trajectory and pseudotime inference with the HSMM dataset
We demonstrate how OCAT infers trajectory and pseudotime using the human skeletal muscle myoblast (HSMM) dataset from Tran and Bader (2020). 

## Table of Contents
- [Step 0. Import data](#data_import)
- [Step 1. Run OCAT](#run_OCAT)
- [Step 2. Trajectory inference](#trajectory)
- [Step 3. Pseudotime inference](#pseudo)


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
Load data

```python
import pandas as pd
from scipy.sparse import csr_matrix

data = pd.read_csv('./HSMM/HSMM.txt', delimiter=' ')
data = data.T
data = csr_matrix(data)
data_list = [data]
```

<a name="run_OCAT"></a>**Step 1. Run OCAT**

Extract the OCAT sparse encoding of the raw gene expression matrix. 

```python
ZW = OCAT.run_OCAT(data_list, m_list=[40], dim=80)
```

<a name="trajectory"></a>**Step 2. Trajectory inference**

Load in the annotated labels.
```python
## load the annotated labels
label = pd.read_csv('./HSMM/HSMM_label.txt', delimiter=' ')
labels_combined_c = np.array(label.loc[:,'V1'])
mapping = {1: 'Fibroblast', 2:'Myotubes', 3: 'Myoblasts', 4:'Undiff', 5:'Intermediates'}
labels_combined = [mapping[i] for i in labels_combined_c]
labels_combined = np.array(labels_combined)
```

`OCAT.compute_lineage()` function infers `Lineages` over clusters with the OCAT features, predicted/true cluster labels and a user-specified `root_cluster`.

```python
Lineage, root_cluster, cluster_labels, tree = OCAT.compute_lineage(ZW, labels_combined, root_cluster='Myoblasts', name='OE', reverse=0)
```
<img src="https://github.com/bowang-lab/OCAT/blob/master/img/trajectory.png" width="350" height="350"/>

<a name="pseudo"></a>**Step 3. Pseudotime inference**

`OCAT.compute_ptime()` function infers pseudotime for individual cell using the OCAT extracted features and the predicted lineage. 
```python
Ptime, root_cell_list = OCAT.compute_ptime(ZW, labels_combined, Lineage, root_cluster=root_cluster, latent=None)
```
<img src="https://github.com/bowang-lab/OCAT/blob/master/img/ptime.png" width="500" height="500"/>
