## Analyzing Spatial Brain scRNA-seq
We demonstrate how OCAT sparsely encodes spatial single-cell gene expression data 

## Table of Contents
- [Step 0. Import data](#data_import)
- [Step 1. Data pre-processing](#pre_processing)
- [Step 2. Dimension reduction](#dim_reduct)
- [Step 3. Contruct bipartite graph through ghost cells](#ghost_cell)
- [Step 4. Clustering \& visualization](#clustering)

```python
import OCAT
import numpy as np
```

<a name="data_import"></a>**Step 0. Import data**     
```python
from scipy.sparse import csr_matrix
from scipy.io import loadmat

my_data = loadmat('./brain_spatial.mat')
in_X = csr_matrix(my_data['A'])
my_data_list = [in_X.T]
```

<a name="pre_processing"></a>**Step 1. Run OCAT**


```python
ZW = OCAT.run_OCAT(my_data_list, m = 125, dim=125)
```

<a name="clustering"></a>**Step 4. Visualization**

<img src="https://github.com/bowang-lab/OCAT/blob/master/vignettes/Spatial/OCAT_spatial_v3.png" width="400" height="400" />  

