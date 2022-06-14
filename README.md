# OCAT - One Cell At A Time
OCAT provides a fast and memory-efficient framework for analyzing and integrating large-scale scRNA-seq data. [Our paper](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02659-1) is now published in Genome Biology!!!

Check out [rOCAT](https://github.com/bowang-lab/rOCAT) to use OCAT in R! 

## :heavy_plus_sign: Method
OCAT constructs sparse representation of cell features through ghost cells in the datasets. These ghost cells serve as bridges to inform on cell-cell similarity between the original cells. With the sparse features extracted, OCAT provides an efficient framework for cell type clustering and dataset integration that achieves state-of-the-art performance.

<br><img src="https://github.com/bowang-lab/OCAT/blob/master/img/Figure1_update.png"/>

## :triangular_ruler: Requirements and Installation
* Linux/Unix
* Python 3.7

Install OCAT package from PyPI. Pre-installation of Numpy and Cython required.
```bash
$ pip install numpy
$ pip install OCAT
```

## :heavy_plus_sign: Tutorials
* [Clustering and Differential Gene Analysis of Mouse Brain scRNA-seq Data (Zeisel et al. 2015)](https://github.com/bowang-lab/OCAT/blob/master/vignettes/Clustering/README.md)
* [Integration of 5 Human Pancreatic scRNA-seq Datasets](https://github.com/bowang-lab/OCAT/blob/master/vignettes/Integration/README.md)
* [Clustering of Spatial scRNA-seq Data](https://github.com/bowang-lab/OCAT/tree/master/vignettes/Spatial/README.md)
* [Trajectory and pseudotime inference using HSMM dataset](https://github.com/bowang-lab/OCAT/blob/master/vignettes/trajectory/README.md)
* [Cell Inference of new incoming data based on reference dataset](https://github.com/bowang-lab/OCAT/blob/master/vignettes/Cell_inference/README.md)
