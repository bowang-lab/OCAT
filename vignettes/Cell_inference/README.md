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
Import the raw gene expression matrix. 

```python
from scipy.io import loadmat
from scipy.sparse import csr_matrix

data = loadmat('./Test_5_Zeisel.mat')
in_X = csr_matrix(data['in_X'])
```
Randomly split the gene expression matrix to 90% reference dataset and 10% inference dataset. 

```python
import random
total_index = list(range(in_X.shape[0]))
random.shuffle(total_index)

ref_index = total_index[:round(0.9*in_X.shape[0])]
inf_index = total_index[round(0.9*in_X.shape[0]):]

ref_data = in_X[ref_index,:]
ref_data_list = [ref_data]

inf_data = in_X[inf_index,:]
inf_data_list = [inf_data]
```

<a name="reference"></a>**Step 1. Run OCAT on the reference dataset**

On the reference dataset, OCAT extracts its sparse encoding and saves the projection of the raw counts to `d`-dimensional subspace. 
```python
ZW_db, db_list = OCAT.run_OCAT(ref_data_list, m_list=[50], dim=30, if_inference=True)
```

<a name="inference"></a>**Step 2. Run OCAT on the inference dataset**

When an inference dataset comes in, it is projected to the same subspace as the reference dataset through `db_list`, and the predicted labels is based on the model trained using the reference dataset. The labels `labels_db` in the training set can be either the annotated cell types (ground truth) or the predicted cell types based on `ZW_db`. Here we adopt the annotated cell type labels. 

```python
labels_true = data['true_labs'].flatten()
labels_combined = labels_true[ref_index]

ZW_inf, labels = OCAT.run_cell_inference(data_list=inf_data_list, ZW_db=ZW_db, labels_db=labels_combined, db_list=db_list)
```

