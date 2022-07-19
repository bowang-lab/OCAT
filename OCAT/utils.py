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
from sklearn.decomposition import PCA, TruncatedSVD
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
import matplotlib.pyplot as plt

def order_genes(data_list, var_list):
    assert len(data_list)==len(var_list), "data_list and var_list length doesn't match"

    idx_list = []
    first_dataset_idx = {j: i for i,j in enumerate(var_list[0])}
    index_intersection = first_dataset_idx.keys()
    idx_list.append(first_dataset_idx)
    
    if len(data_list)>1:
        for var in var_list[1:]:
            index_dict = {j: i for i,j in enumerate(var)}
            idx_list.append(index_dict)
            index_intersection = index_intersection & index_dict.keys()

    assert len(index_intersection)>0, "No Common elements found in var_list"

    tmp_data_list = []

    for i, dataset in enumerate(data_list):
        idx = [idx_list[i].get(var) for var in index_intersection]
        tmp = scipy.sparse.csr_matrix(dataset[idx,:])
        tmp_data_list.append(tmp)
    return tmp_data_list,[list(index_intersection)]*len(var_list)

def normalize_data(data_list, is_memory=True):
    for i, X in enumerate(data_list):
        if is_memory:
            X.data = np.log10(X.data+1).astype(np.float32)
            #X.data = X.data.astype(np.float32)
        else:
            X = np.log10(X+1).astype(np.float32)
            X = np.ascontiguousarray(X)
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

def TFIDF(data, type, scale_factor=10000):
    data = data.T
    nCells = data.shape[1]
    nPeaks = data.shape[0]
    peak_sum = data.sum(axis=1)
    lib_size = data.sum(axis=0)
    avg_peak_signal = peak_sum/nCells
    if type==1:
        tf = data.multiply(1/lib_size)
        idf = 1/avg_peak_signal
        norm_data = tf.multiply(idf)
        return (norm_data*scale_factor).log1p()
    elif type==2:
        tf = data.multiply(1/lib_size)
        idf = 1/avg_peak_signal
        idf = idf.log1p()
        return tf.multiply(idf)
    elif type==3:
        tf = data.multiply(1/lib_size)
        tf = tf*scale_factor.log1p()
        idf = 1/avg_peak_signal
        idf = idf.log1p()
        return tf.multiply(idf)
    elif type==4:
        idf = 1/avg_peak_signal
        return data.multiply(idf)
    else:
        return data

def preprocess(data_list, log_norm, l2_norm, tfidf=0):
    assert len(data_list) > 0, "Data list cannot be empty"
    assert tfidf in [0,1,2,3,4], "tfidf can only be one of 0,1,2,3,4"
    # Check data format, must be sparse.csr_matrix or np.ndarray
    if isinstance(data_list[0], scipy.sparse.csr_matrix):
        is_memory=True
    elif isinstance(data_list[0], np.ndarray):
        is_memory=False
    else:
        sys.exit("Data matrix must be np.ndarray or sparse.csc_matrix")
    # Perform normalization
    if tfidf:
        data_list = [TFIDF(i, type=tfidf).T for i in data_list]
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

def m_estimate(data_list):
    m_list = []
    for i in data_list:
        n_cells = i.shape[0]
        if n_cells < 2000:
            m_list.append(20)
        elif n_cells < 10000:
            m_list.append(40)
        else:
            m_list.append(60)
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
#     mode                           -- 'FSM' or 'TruncatedSVD'
# Out:
#     data_list   [(a,dim)...(z,dim)]    -- list of datasets
# Description: The function reduces the datasets to dim subspaces
##############################################
def apply_dim_reduct(data_list, dim=None, mode='FSM', random_seed=42, upsample=False):
    assert len(data_list) > 0, 'Data list cannot be empty'
    assert mode in ['FSM', 'TruncatedSVD'], 'Select dimension reduction method: FSM or TruncatedSVD'
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
        return data_list, Wm
    elif mode == 'TruncatedSVD':
        svd = TruncatedSVD(n_components=dim, n_iter=15, random_state=42)
        svd.fit(data_combined)
        data_list = [np.nan_to_num(svd.transform(i)) for i in data_list]
    return data_list

def apply_dim_reduct_inference(data_list, Wm):
    data_list = [i.dot(Wm) for i in data_list]
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

def evaluate_clusters(Z, num_cluster=None, n_init=20, return_num_cluster=False):
    if num_cluster is None:
        cluster_label = estimate_num_cluster(Z)
        num_cluster = len(np.unique(cluster_label))
    clusters = KMeans(n_clusters=num_cluster, n_init=n_init).fit(Z)
    if return_num_cluster:
        return clusters.labels_, num_cluster
    else:
        return clusters.labels_

def plot_umap(Z, labels, show_plot=True):
    reducer = umap.UMAP()
    Z_scaled = StandardScaler().fit_transform(Z)
    embedding = reducer.fit_transform(Z_scaled)
    if show_plot:
        plt.figure()
        ax = sns.scatterplot(embedding[:, 0], embedding[:, 1], hue=labels, palette=sns.color_palette('muted', n_colors=len(np.unique(labels))))
        ax.set(xlabel='UMAP0', ylabel='UMAP1', xticklabels=[], yticklabels=[])
        ax.set_xticks([])
        ax.set_yticks([])
        return ax.get_figure(),embedding
    return embedding