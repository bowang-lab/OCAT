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

labels_true = data['true_labs'].flatten()
labels_combined = labels_true[ref_index]
```

<a name="reference"></a>**Step 1. Run OCAT on the reference dataset**

On the reference dataset, OCAT extracts its sparse encoding and saves the projection of the raw counts to `d`-dimensional subspace. 

In our tool, there are two type of reference building technique:
1. the groud truth labels for reference data_liist are known
2. the groud truth labels for reference data_liist are unknown

It is differentiated by a `labels_true` parameter, which intakes a list of true labels for all reference datasets in the `data_list`. Ignore it if there is no groud truth labels. 

```python
ZW_db, db_list =  OCAT.run_OCAT(ref_data_list, dim=30, if_inference=True, labels_true=[labels_combined])
```

<a name="inference"></a>**Step 2. Run OCAT on the inference dataset**

When an inference dataset comes in, it is projected to the same subspace as the reference dataset through `db_list`, and the predicted labels is based on the model trained using the reference dataset. The labels `labels_db` in the training set can be either the annotated cell types (ground truth) or the predicted cell types based on `ZW_db`. Here we adopt the annotated cell type labels. 

If the `labels_db` is the groud truth labels, please make parapmeter `true_known=True`

1. When `labels_db` is the groud truth labels:
```python
ZW_inf, labels = OCAT.run_cell_inference(data_list=inf_data_list, db_list=db_list, true_known=True, labels_db=labels_combined)
```

2. When `labels_db` is the predicted labels:
```python
ZW_inf, labels = OCAT.run_cell_inference(data_list=inf_data_list, db_list=db_list, ZW_db= ZW_db, true_known=False, labels_db=labels_combined)
```

