## Analyzing Spatial Brain scRNA-seq
We demonstrate how OCAT sparsely encodes spatial single-cell gene expression data 

## Table of Contents
- [Step 0. Import data](#data_import)
- [Step 1. Run OCAT](#run_OCAT)
- [Step 2. Clustering \& visualization](#clustering)

```python
import OCAT
import numpy as np
```

<a name="data_import"></a>**Step 0. Import data**     
```python
from scipy.sparse import csc_matrix

my_data = np.load('brain_spatial.npz')
in_X = csc_matrix(my_data['data'])
my_data_list = [in_X.T]
```

<a name="run_OCAT"></a>**Step 1. Run OCAT**


```python
ZW = OCAT.run_OCAT(my_data_list, m_list = [125], dim=125)
```

<a name="clustering"></a>**Step 2. Clustering \& Visualization**

```python
from sklearn.cluster import KMeans

pca_ground_ZW = KMeans(n_clusters=15, n_init=20).fit(ZW)
labels_pred = pca_ground_ZW.labels_
```

<img src="https://github.com/bowang-lab/OCAT/blob/master/vignettes/Spatial/OCAT_spatial_v3.png" width="400" height="400" />  

