import numpy as np
import os
from .fast_similarity_matching import FSM
import faiss
from scipy import *
from scipy.sparse import *
import scipy.sparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA, KernelPCA
import csv
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from scipy.sparse import *
import scipy.sparse
warnings.filterwarnings('ignore')
import random
import pandas as pd
import sys
#import FSM as FSM_example
#from memory_profiler import profile
import sys
#import umap
import umap.umap_ as umap
import seaborn as sns
from .lineage import estimate_num_cluster

def normalize_data(data_list, is_memory=True):
    for i, X in enumerate(data_list):
        if is_memory:
            X.data = np.log10(X.data+1).astype(np.float32)
            #X.data = X.data.astype(np.float32)
        else:
            X = np.log10(X+1).astype(np.float32)
            X = np.ascontiguousarray(X)
        print(data_list[i].shape)
    return data_list

def l2_normalization(data_list, is_memory=True):
    for i, X in enumerate(data_list):
        if is_memory:
            X = X.tocsc()
        X = normalize(X, norm='l2', axis=0, copy=False)
        if is_memory:
            X = X.tocsr()
        else:
            X = np.ascontiguousarray(X)
        data_list[i] = X
    return data_list

def preprocess(data_list, log_norm, l2_norm):
    assert len(data_list) > 0, "Data list cannot be empty"
    # Check data format, must be sparse.csr_matrix or np.ndarray
    if isinstance(data_list[0], scipy.sparse.csr_matrix):
        is_memory=True
    elif isinstance(data_list[0], np.ndarray):
        is_memory=False
    else:
        sys.exit("Data matrix must be np.ndarray or sparse.csc_matrix")
    # Perform normalization
    if log_norm:
        data_list = normalize_data(data_list, is_memory)
    if l2_norm:
        data_list = l2_normalization(data_list, is_memory)
    return data_list

##############################################
#     k      ()     -- Dimension of PCA subspace to learn
#     Minv0 (k,k)   -- Initial guess for the inverse lateral weight matrix M
#     Uhat0 (k,d)   -- Initial guess for the forward weight matrix W
# Out:
#     Wm    (k,d)   --
#     Minv  (k,k)   -- The inverse lateral weight matrix M
#     W     (k,d)   -- The forward weight matrix W
# Description: The function estimates the top K dimensional principal subspace for X
#     yt <- xt * Minv * W
##############################################
def online_FSM(X, k, random_seed, Minv0 = None, Uhat0 = None, scal = 10, ind = None):
    np.random.seed(random_seed)
    n, d = X.shape
    k = min(k, d)

    if Minv0 is not None and Uhat0 is not None:
        pass
    else:
        # initialize Uhat and Minv
        Uhat0 = np.random.normal(0, 1, (k, d)) / scal
        Uhat0 = Uhat0.astype(np.float64)
        Minv0 = np.identity(k, dtype=np.float64) * scal

    fsm = FSM(k, d, Minv0, Uhat0)
    if ind is None:
        ind = np.random.randint(n,size=n)
    for i in ind:
        xx = np.array(X[i,:]).flatten()
        fsm.fit_next(xx)

    Wm = np.matmul(fsm.Minv,fsm.W).transpose()
    Minv = fsm.Minv
    W = fsm.W
    return Wm, Minv, W

##############################################
#     k      ()     -- Dimension of PCA subspace to learn
#     Minv0 (k,k)   -- Initial guess for the inverse lateral weight matrix M
#     Uhat0 (k,d)   -- Initial guess for the forward weight matrix W
# Out:
#     Wm    (k,d)   --
#     Minv  (k,k)   -- The inverse lateral weight matrix M
#     W     (k,d)   -- The forward weight matrix W
# Description: The function estimates the top K dimensional principal subspace for X
#     yt <- xt * Minv * W, with csr sparse data
##############################################
def online_FSM_sp(X, k, random_seed, Minv0 = None, Uhat0 = None, scal = 10, ind=None):
    np.random.seed(random_seed)
    n, d = X.shape
    k = min(k, d)

    if Minv0 is not None and Uhat0 is not None:
        pass
    else:
        # initialize Uhat and Minv
        Uhat0 = np.random.normal(0, 1, (k, d)) / scal
        Minv0 = np.identity(k) * scal

    fsm = FSM(k, d, Minv0, Uhat0)
    if ind is None:
        ind = np.random.randint(n,size=n)
    t = 0
    W = Uhat0
    Minv = Minv0
    outer_W = np.empty_like(W)
    outer_Minv = np.empty_like(Minv)
    for i in ind:
        xx = np.array(X.getrow(i).todense()).flatten()
        fsm.fit_next(xx)
    Wm = np.matmul(fsm.Minv,fsm.W).transpose()
    Minv = fsm.Minv
    W = fsm.W
    return Wm, Minv, W

#TODO
def dim_estimate(data_list):
    gene_num = data_list[0].shape[1]
    if gene_num < 5000:
        dim = 50
    elif gene_num < 10000:
        dim = 100
    else:
        dim = 125
    print('estimated dim: {}'.format(dim))
    return dim

#TODO
def m_estimate_(data_list):
    min_cell_num = min([i.shape[0] for i in data_list])
    m = max(20, round(min_cell_num/100))
    print('estimated m: {}'.format(m))
    return m

def m_estimate(data_list):
    m_list = [max(20, round(i.shape[0]/100)) for i in data_list]
    print('estimated m_list: {}'.format(m_list))
    return m_list

def balanced_smapling(data_list, random_seed=42):
    n_dataset = [X.shape[0] for X in data_list]
    n_idx = np.max(n_dataset)
    n_dataset = np.cumsum(np.array(n_dataset))
    n_dataset = np.insert(n_dataset, 0, 0)
    #n_idx = int(n_dataset[-1]/(len(n_dataset)-1))
    ind_balanced = []
    np.random.seed(random_seed)
    for i in range(len(n_dataset)-1):
        ind = np.random.randint(n_dataset[i], n_dataset[i+1], n_idx)
        ind_balanced.append(ind)
    ind_balanced = np.concatenate(ind_balanced)
    np.random.shuffle(ind_balanced)
    return np.array(ind_balanced)

##############################################
# In: data_list   [(a,n)...(z,n)]    -- list of datasets
#     dim                            -- desired dimension after dimensionality reduction
#     m                              -- num of anchors
# Out:
#     data_list   [(a,dim)...(z,dim)]    -- list of datasets
# Description: The function reduces the datasets to dim subspaces
##############################################
def apply_dim_reduct(data_list, dim=None, mode='FSM', random_seed=42, upsample=False):
    assert len(data_list) > 0, 'Data list cannot be empty'
    assert mode in ['FSM', 'pca'], 'Select dimension reduction method: FSM or PCA'
    # Check data format, must be sparse.csr_matrix or np.ndarray
    if isinstance(data_list[0], scipy.sparse.csr_matrix):
        is_memory=True
    elif isinstance(data_list[0], np.ndarray):
        is_memory=False
    else:
        sys.exit('Data matrix must be np.ndarray or sparse.csc_matrix')
    # Compute default values of dim and m if not already specified
    if dim==None:
        dim = dim_estimate(data_list)
    if is_memory:
        data_combined = scipy.sparse.vstack(data_list)
    else:
        data_combined = np.concatenate(data_list, axis=0)
    if mode == 'FSM':
        if is_memory:
            if upsample:
                ind = balanced_smapling(data_list, random_seed=random_seed)
                Wm, _ , _ = online_FSM_sp(data_combined, dim, random_seed, ind=ind)
            else:
                Wm, _ , _ = online_FSM_sp(data_combined, dim, random_seed)
            data_list = [i.dot(Wm) for i in data_list]
        else:
            if upsample:
                ind = balanced_smapling(data_list, random_seed=random_seed)
                Wm, _ , _ = online_FSM(data_combined, dim, random_seed, ind=ind)
            else:
                Wm, _ , _ = online_FSM(data_combined, dim, random_seed)
            data_list = [np.matmul(i, Wm) for i in data_list]
    else:
        #TODO: test sparse implementation
        pca = KernelPCA(n_components=dim, kernel='cosine', eigen_solver='arpack', random_state=random_seed)
        pca.fit(data_combined)
        data_list = [np.nan_to_num(pca.transform(i)) for i in data_list]
        anchor_list = [np.nan_to_num(pca.transform(i)) for i in anchor_list]
    return data_list

##############################################
# In: Z                              -- OCAT features
#     num_cluster                    -- num of clusters desired
# Out:
#     pca_ground.labels_             -- list of predicted cluster labels
# Description: The function reduces the datasets to dim subspaces
##############################################
def evaluate_clusters_(Z, num_cluster, n_init=20, return_umap=True):
    clusters = KMeans(n_clusters=num_cluster, n_init=n_init).fit(Z)
    if return_umap:
        reducer = umap.UMAP()
        Z_scaled = StandardScaler().fit_transform(Z)
        embedding = reducer.fit_transform(Z_scaled)
        ax = sns.scatterplot(embedding[:, 0], embedding[:, 1], hue=clusters.labels_, palette=sns.color_palette('muted', n_colors=num_cluster))
        ax.set(xlabel='UMAP0', ylabel='UMAP1', xticklabels=[], yticklabels=[])
        ax.set_xticks([])
        ax.set_yticks([])
        return clusters.labels_, ax.get_figure()
    else:
        return clusters.labels_

def evaluate_clusters(Z, num_cluster=None, n_init=20):
    if num_cluster is None:
        cluster_label = estimate_num_cluster(Z)
    num_cluster = len(np.unique(cluster_label))
    clusters = KMeans(n_clusters=num_cluster, n_init=n_init).fit(Z)
    return clusters.labels_

def plot_umap(Z, labels):
    reducer = umap.UMAP()
    Z_scaled = StandardScaler().fit_transform(Z)
    embedding = reducer.fit_transform(Z_scaled)
    ax = sns.scatterplot(embedding[:, 0], embedding[:, 1], hue=labels, palette=sns.color_palette('muted', n_colors=len(np.unique(labels))))
    ax.set(xlabel='UMAP0', ylabel='UMAP1', xticklabels=[], yticklabels=[])
    ax.set_xticks([])
    ax.set_yticks([])
    return ax.get_figure()
