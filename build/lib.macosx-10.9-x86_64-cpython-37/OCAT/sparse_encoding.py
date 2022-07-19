import numpy as np
import pandas as pd
import os
import faiss
from .fast_similarity_matching import FSM
from .utils import m_estimate, dim_estimate, apply_dim_reduct, apply_dim_reduct_inference, preprocess, evaluate_clusters
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import OCAT.example as example
from sklearn import svm

##############################################
# In: data_list   [(a,n)...(z,n)]    -- list of datasets
#     m                              -- num of anchors
# Out:
#     anchor list [(m,n)...(m,n)]    -- list of anchors 
# Description: The function generates the corresponding anchors for the datasets
##############################################
def find_anchors(data_list, m):
    anchor_list = []
    for i, X in enumerate(data_list):
        X = X.astype(np.float32)
        d = X.shape[1]
        kmeans = faiss.Kmeans(d, m[i], niter=20, verbose=False)
        kmeans.train(X)
        anchors = kmeans.centroids
        anchor_list.append(anchors)
    return anchor_list

##############################################
# In: X      (c,n)
# Out:
#     S     (c,n)
# Description: The function sorts data in descending order and
#      picks the first kk th that sums up to 1 and scales them
#      to be of range 0 to 1
##############################################
def SimplexPr(X):
    C, N = X.shape
    #to sort in descending order
    T = np.sort(X,axis=0) [::-1]
    S = np.array(X).astype(np.float)
    for i in range(N):
        kk = -1
        t = T[:,i]
        #find kk where first kk element >= 1 otherwise kk = last index
        for j in range(C):
            tep = t[j] - (np.sum(t[0:j+1]) - 1)/(j+1)
            if tep <= 0:
                kk = j-1
                break
        if kk == -1:
            kk = C-1
        #scale X to be at 1
        theta = (np.sum(t[0:kk+1]) - 1)/(kk+1)
        #theta = np.expand_dims(theta, axis=1)
        S[:,i] = (X[:,i] - theta).clip(min=0).flatten()
        #X = np.subtract(X, theta.clip(min=0))
    return S

####################################################################
# In: X      (d,1)  -- data from one cell
#     U      (d,s)  -- Anchor data, s is the # of the closest anchors
#     cn     ()     -- # of iterations, 5-20
# Out:
#     z     (s,1)   --
# Description: This function updates z cn times to find the best z
#      such that x <- U * z
#
###################################################################
def LAE (x,U,cn):
    d, s = U.shape
    #print("d, s: ", d, s)
    z0 = np.ones((s,1))/s
    z1 = z0
    delta = np.zeros((1,cn+2))
    delta[0][0] = 0
    delta[0][1] = 1
    beta = np.zeros((1,cn+1))
    beta[0][0] = 1
    for t in range(cn):
        alpha = (delta[0][t]-1)/delta[0][t+1]
        v = z1 + alpha*(z1-z0)
        dif = x - np.matmul(U,v)
        gv = np.matmul(dif.transpose(),dif/2)
        dgv = np.matmul(U.transpose(),np.matmul(U,v)-x)
        for j in range(d+1):
            b = 2**j*beta[0][t]
            z = SimplexPr(v-dgv/b)
            dif = x - np.matmul(U,z)
            gz = np.matmul(dif.transpose(),dif/2)
            dif = z - v
            gvz = gv + np.matmul(dgv.transpose(),dif) + b * np.matmul(dif.transpose(),dif/2)
            if gz <= gvz:
                beta[0][t+1] = b
                z0 = z1
                z1 = z
                break
        if beta[0][t+1] == 0:
            beta[0][t+1] = b
            z0 = z1
            z1 = z
        delta[0][t+2] = ( 1+np.sqrt(1+4*delta[0][t+1]**2) )/2
        if np.sum(abs(z1-z0)) <= 1e-4:
            break
    z = z1
    return z, np.sum(abs(z1-z0))

####################################################################
# In: TrainData  (d,n)  -- input data matrix, d dimension, n # of samples
#     Anchor     (d,m)  -- Anchor data, m # of anchors
#     s          ()     -- # of closest anchors
#     flag       ()     -- 0 gives a Gaussian kernel-defined Z and 1 gives a LAE-optimized Z
#     cn         ()     -- # of iterations for LAE, usually set to 5-20
# Out:
#     Z          (n,m)  -- anchor-to-data regression weight matrix
#     rL         (m,m)  -- reduced graph Laplacian matrix
#
###################################################################
def AnchorGraph (TrainData, Anchor, s, flag, cn = 5):
    d,m = Anchor.shape
    _, n = TrainData.shape
    Similarity = cosine_similarity(TrainData.transpose(), Anchor.transpose())
    val = np.sort(Similarity, axis=1)[:, -s:]
    pos = np.argsort(Similarity, axis=1)[:, -s:]
    #flatten matrix and reshape back to nxm
    ind = ((pos)*n + np.repeat(np.array([range(n)]).transpose(),s,1)).astype(int)
    # Gaussian kernel-defined Z
    if flag == 0:
        sigma = np.mean(np.sqrt(val[:,s-1]))
        val = np.exp(-val/(sigma*sigma))
        val = np.repeat(np.array([1/np.sum(val,1)]).transpose(),s,1)*val
    #LAE-optimized Z
    if flag == 1:
        val = val/np.expand_dims(np.sum(val,1), axis=1)
    else:
        Anchor = np.matmul(Anchor,np.diag(1/np.sqrt(np.sum(Anchor*Anchor,axis=0))))
        for i in range(n):
            x = TrainData[:,i]
            x = (x/np.linalg.norm(x,2)).reshape((len(x),1))
            U = Anchor[:,pos[i,:]]
            a = example.LAE(x,U,cn)
            val[i,:] = a[0].flatten()
    #expand s # closest anchors to m
    Z = np.zeros((n* m))
    Z[ind] = [val]
    Z = Z.reshape((m,n)).transpose().astype(np.float32)
    #Z = np.matmul(Z, np.diag(np.sqrt(np.sum(Z,0)**-1)))
    return Z

####################################################################
# In:   Z          (n,m)  -- normalized anchor-to-data regression weight matrix
#       zW         (n,m)  -- unnormalized anchor-to-data regression weight matrix
# Out:  ZW         (n,m)  -- approximation for ZW, where W=(n,n) similarity matrix
###################################################################
def Z_to_ZW(Z):
    W_anchor = np.matmul(Z.T, Z)
    W_diag = np.diag(np.sqrt(np.sum(W_anchor,0)**-1))
    W_anchor = np.linalg.multi_dot([W_diag, W_anchor, W_diag])
    ZW = np.matmul(Z, W_anchor)
    return ZW, W_anchor

####################################################################
# In:   Z          (n,m)  -- input regression weight matrix
# Out:
#       Z          (n,m)  -- matrix norm normalized regression weight matrix
###################################################################
def norm(Z):
    Z_norm = np.linalg.norm(Z, axis=1)
    Z_norm = np.expand_dims(Z_norm, axis=1)
    Z = np.divide(Z, Z_norm)
    Z = np.nan_to_num(Z)
    return Z

####################################################################
# In:   data_list       [(a,dim)...(z,dim)]    -- list of datasets (dim PCs)
#       m_list                                 -- num of anchors
#       s_list                                 -- num of anchors to be selected
#       p                                      -- percentage of NNs to consider
#       cn                                     -- rounds of optimization
#       if_inference                           -- flag for cell inference
#       true_known                             -- flag for true labels known
# Out:  ZW              (a+...+z, m)           -- OCAT feature matrix
###################################################################
def sparse_encoding_integration(data_list, m_list, s_list=None, p=0.3, cn=5, if_inference=True, true_known=False):
    # find anchors
    anchor_list = find_anchors(data_list, m_list)
    if true_known:
        anchor_list = [np.concatenate(anchor_list,axis=0)]
    # construct sparse anchor graph
    Z_list = []
    for i, dataset in enumerate(data_list):
        dataset_Z_list = []
        for j, anchor in enumerate(anchor_list):
            Z = AnchorGraph(dataset.transpose(), anchor.transpose(), s_list[j], 2, cn)
            dataset_Z_list.append(Z)
        dataset_Z = np.concatenate(dataset_Z_list, axis=1)
        Z_list.append(dataset_Z)
    Z = np.nan_to_num(np.concatenate(Z_list, axis=0))
    ZW, W_anchor = Z_to_ZW(Z)
    ZW = norm(np.nan_to_num(ZW))
    if if_inference:
        return ZW, anchor_list, s_list, W_anchor
    else:
        return ZW

####################################################################
# In:   data_list       [(a,dim)...(z,dim)]    -- list of datasets (dim PCs)
#       m_list                                 -- num of anchors
#       s_list                                 -- num of anchors to be selected
#       dim                                    -- num of dimensions after dim reduct
#       p                                      -- percentage of NNs to consider
#       log_norm                               -- if apply log norm
#       l2_norm                                -- if apply l2 norm
#       if_inference                           -- if prepare for cell inference
#       random_seed                            -- random seed
#       labels_true                            -- the true labels for each dataset in the data_list
# Out:  ZW              (a+...+z, m)           -- OCAT feature matrix
#       db_list                                -- anchor_list, s_list, W_anchor, Wm
#                                                 from reference dataset for cell inference
###################################################################
def run_OCAT(data_list, m_list=None, s_list=None, dim=None, p=0.3, log_norm=True, l2_norm=True, tfidf=0, mode='FSM', if_inference=False, random_seed=42, labels_true=None):
    if m_list == None:
        m_list = m_estimate(data_list)
    if s_list ==None:
        s_list = [round(p*m) for m in m_list]
    if dim == None:
        dim = dim_estimate(data_list)

    data_list = preprocess(data_list, log_norm=log_norm, l2_norm=l2_norm, tfidf=tfidf)
    if if_inference:
        data_list, Wm = apply_dim_reduct(data_list, dim=dim, mode=mode, random_seed=random_seed)
        true_known = False
        if labels_true:
            true_known = True
            data_combined = np.concatenate(data_list, axis=0)
            labels_true_combined = np.concatenate(labels_true, axis=0)
            data_list = [data_combined[labels_true_combined==i,:] for i in np.unique(labels_true_combined)]
            m_sum = np.sum(m_list)
            # extract the cell numbers in each true cluster
            m_list = [data_list[i].shape for i in range(len(np.unique(labels_true_combined)))]
            # assign m based on the cell proportion
            total_cell_num = data_combined.shape[0]
            m_list = [int(m_sum*i[0]/total_cell_num) if int(m_sum*i[0]/total_cell_num)>1 else 1 for i in m_list]

            s_list = [round(p*m) for m in m_list]
            s_list = [np.sum(s_list)]
            print('New m_list based on true cell type cluster: ',m_list)

        ZW, anchor_list, s_list, W_anchor = sparse_encoding_integration(data_list, m_list=m_list, s_list=s_list, p=p, cn=5, if_inference=True, true_known=true_known)
        if labels_true: 
            db_list = [anchor_list, s_list, W_anchor, Wm, m_list] 
        else:
            db_list = [anchor_list, s_list, W_anchor, Wm]
        return ZW, db_list
    else:
        data_list, _ = apply_dim_reduct(data_list, dim=dim, mode=mode, random_seed=random_seed)
        ZW = sparse_encoding_integration(data_list, m_list=m_list, s_list=s_list, p=p, cn=5, if_inference=False)
        return ZW

####################################################################
# In:   data_list       [(a,dim)...(z,dim)]    -- list of inference datasets (dim PCs)
#       labels_db                              -- cell type annotations from reference dataset
#       db_list                                -- reference db info returned from run_OCAT
#       true_known                             -- if the true labels for reference db is known
#       ZW_db                                  -- OCAT features of the reference dataset
#       log_norm                               -- if apply log norm
#       l2_norm                                -- if apply l2 norm
#       cn                                     -- rounds of optimization
# Out:  ZW              (a+...+z, m)           -- OCAT features of the inference dataset
#       labels                                 -- inferred cell type labels from inference dataset
###################################################################
def run_cell_inference(data_list, labels_db, db_list, true_known=False, ZW_db=list(), log_norm=True, l2_norm=True, cn=5):

    if true_known:
        [anchor_list, s_list, W_anchor, Wm, m_list] = db_list
    else:
        assert ZW_db!=list(), "Must input ZW_db if the reference datasets has no true labels"
        [anchor_list, s_list, W_anchor, Wm] = db_list

    data_list = preprocess(data_list, log_norm=log_norm, l2_norm=l2_norm)
    data_list = apply_dim_reduct_inference(data_list, Wm)
    Z_list = []
    for i, dataset in enumerate(data_list):
        dataset_Z_list = []
        for j, anchor in enumerate(anchor_list):
            Z = AnchorGraph(dataset.transpose(), anchor.transpose(), s_list[j], 2, cn)
            dataset_Z_list.append(Z)
        dataset_Z = np.concatenate(dataset_Z_list, axis=1)
        Z_list.append(dataset_Z)
    Z = np.nan_to_num(np.concatenate(Z_list, axis=0))

    if true_known:
        all_index = np.cumsum([0]+ m_list)
        anchor_num_index = list(zip(all_index[:len(all_index)-1], all_index[1:]))
        ## add up all the weights for the each true cluster and choose the cluster with max as assigned label
        labels = np.argmax([[np.sum(data[x:y]) for x,y in anchor_num_index] for data in Z],axis=1)
        labels = np.unique(labels_db)[labels]
        return Z, labels
    else:
        ZW = np.matmul(Z, W_anchor)
        ZW = norm(np.nan_to_num(ZW))

        clf = SVC(random_state=42)
        clf.fit(ZW_db, labels_db)
        labels = clf.predict(ZW)
    return ZW, labels

def post_processing_pca(Z, topk=20):
    # center data by standard scaling
    scaler=StandardScaler()
    scaler.fit(Z)
    Z = scaler.transform(Z)
    # take topk PCs
    pca = PCA(n_components=topk, svd_solver='arpack')
    pca_result = pca.fit_transform(Z)
    return pca_result

def tune_hyperparameters(data_list, if_tune_m=True, m_range=None, if_tune_dim=True, dim_range=None, if_tune_p=False, p_range=None, log_norm=True, l2_norm=True, true_labels=None, verbose=True):
    # Specify data normalization
    data_list = preprocess(data_list, log_norm=log_norm, l2_norm=l2_norm)
    num_datasets = len(data_list)
    # Impute m if None
    if m_range==None:
        m_est = max(m_estimate(data_list))
        if if_tune_m:
            m_range = [m_est+i*5 for i in range(-3, 3)]
        else:
            m_range = [m_est]
            print('WARNING no value of m is given, default m={} for the dataset(s) from estimation.'.format(m_est))
    # Impute dim if None
    if dim_range==None:
        dim_est = dim_estimate(data_list)
        if if_tune_dim:
            dim_range = [dim_est+i*10 for i in range(-2, 2)]
        else:
            dim_range = [dim_est]
            print('WARNING no value of dim is given, default dim={} for the dataset(s) from estimation.'.format(dim_est))
    # Impute p if None
    if p_range==None:
        if if_tune_p:
            p_range = [0.1, 0.3, 0.5]
        else:
            p_range = [0.3]
            print('WARNING no value of p is given, default p=0.3 for the dataset(s) from estimation.')
    # If ground truth given, find n_clusters
    if true_labels is not None:
        n_clusters = len(np.unique(true_labels))
    out = []
    if verbose:
        print('Testing hyperparameters in the range below:')
        print('Range for m: {}'.format(m_range))
        print('Range for dim: {}'.format(dim_range))
        print('Range for p: {}'.format(p_range))
    for m in m_range:
        for n_dim in dim_range:
            for p in p_range:
                if m*p < 3:
                    print('Skip m={} and p={} as the number of ghost cells is smaller than 3.'.format(m, p))
                    continue
                ZW = run_OCAT(data_list=data_list, m_list=[m]*num_datasets, dim=n_dim, p=p, log_norm=False, l2_norm=False)
                if true_labels is None:
                    labels_pred, n_clusters = evaluate_clusters(ZW, return_num_cluster=True)
                    sil_score = silhouette_score(ZW, labels_pred)
                    out.append([m, n_dim, p, n_clusters, sil_score])
                else:
                    labels_pred = evaluate_clusters(ZW, num_cluster=n_clusters)
                    NMI_cell = normalized_mutual_info_score(true_labels, labels_pred)

                    AMI_cell = adjusted_mutual_info_score(true_labels, labels_pred)

                    ARI_cell = adjusted_rand_score(true_labels, labels_pred)
                    out.append([m, n_dim, p, NMI_cell, AMI_cell, ARI_cell])
    out = np.array(out)
    if true_labels is not None:
        df = pd.DataFrame(data=out, columns=['m', 'n_dim', 'p', 'NMI_score', 'AMI_score', 'ARI_score'])
    else:
        df = pd.DataFrame(data=out, columns=['m', 'n_dim', 'p', 'n_clusters', 'silhoutte_score'])
    if verbose:
        print(df)
    return df
