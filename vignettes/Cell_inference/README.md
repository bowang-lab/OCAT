## Cell inference using the mouse cortex dataset
OCAT supports immediate cell type inference of incoming data based on existing databases, without re- computing the latent representations by combining the new incoming ("inference") dataset and the existing ("reference") dataset.

## Table of Contents
- [Step 0. Import data](#data_import)
- [Step 1. Run OCAT on the reference dataset](#reference)
- [Step 2. Run OCAT on the inference dataset](#inference)

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

