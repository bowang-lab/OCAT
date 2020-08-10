# OCAT - One Cell At A Time
OCAT provides a fast and memory-efficient framework for large-scale scRNA-seq data. 

## :heavy_plus_sign: Method
OCAT constructs sparse representation of cell features through ghost cells in the datasets. These ghost cells serve as bridges to inform on cell-cell similarity between the original cells. With the sparse features extracted, OCAT provides an efficient framework for cell type clustering and dataset integration that achieves state-of-the-art performance.

![Alt text](https://github.com/bowang-lab/OCAT/blob/master/img/Figure_1_August9.png)

## :triangular_ruler: Requirements and Installation
* Linux/Unix
* Python 3.7

Install OCAT package from PyPI.
```bash
$ pip install OCAT
```

## :heavy_plus_sign: Notebooks and Tutorials
* [Clustering of Mouse Brain scRNA-seq Data (Zeisel et al. 2015)](https://github.com/bowang-lab/OCAT/blob/master/vignettes/Clustering/README.md)
* [Integration of 5 Human Pancreatic scRNA-seq Datasets](https://github.com/bowang-lab/OCAT/blob/master/vignettes/Integration/README.md)
