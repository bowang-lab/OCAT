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

<a name="reference"></a>**Step 1. Run OCAT on the reference dataset**


```python
ZW_db, db_list = OCAT.run_OCAT(data_list, m_list=[50], s_list=None, dim=30, p=0.3, log_norm=True, l2_norm=True, if_inference=True, random_seed=42)
```

<a name="inference"></a>**Step 2. Run OCAT on the inference dataset**

```python
ZW, labels = OCAT.run_cell_inference(data_list, ZW_db, labels_combined, db_list)
ZW = OCAT.run_OCAT(data_list, m_list=[50], s_list=None, dim=30, p=0.3, log_norm=True, l2_norm=True, if_inference=False, random_seed=42)

out = OCAT.evaluate_clusters(ZW)
```

<img src="https://github.com/bowang-lab/OCAT/blob/master/vignettes/Spatial/OCAT_spatial_v3.png" width="400" height="400" />  

