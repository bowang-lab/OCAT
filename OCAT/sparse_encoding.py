import numpy as np
import os
import faiss
from .fast_similarity_matching import FSM
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import OCAT.example as example
from .utils import m_estimate

##############################################
# In: data_list   [(a,n)...(z,n)]    -- list of datasets
#     m                              -- num of anchors
# Out:
#     anchor list [(m,n)...(m,n)]    -- list of anchors 
# Description: The function generates the corresponding anchors for the datasets
##############################################
def find_anchors_(data_list, m):
    anchor_list = []
    for X in data_list:
        X = X.astype(np.float32)
        n,d = X.shape
        kmeans = faiss.Kmeans(d, m, niter=20, verbose=False)
        kmeans.train(X)
        anchors = kmeans.centroids
        anchor_list.append(anchors)
    return anchor_list

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
        #print("alpha: ", alpha.shape)
        v = z1 + alpha*(z1-z0)
        #print("v: ", v.shape)
        dif = x - np.matmul(U,v)
        #print("dif: ", dif.shape)
        gv = np.matmul(dif.transpose(),dif/2)
        #print("gv: ", gv.shape)
        dgv = np.matmul(U.transpose(),np.matmul(U,v)-x)
        #print("dgv: ", dgv.shape)
        for j in range(d+1):
            b = 2**j*beta[0][t]
            #print("b: ", b.shape)
            #TODO
            z = SimplexPr(v-dgv/b)
            #print("z: ", z.shape)
            dif = x - np.matmul(U,z)
            gz = np.matmul(dif.transpose(),dif/2)
            #print("gz: ", gz.shape)
            dif = z - v
            gvz = gv + np.matmul(dgv.transpose(),dif) + b * np.matmul(dif.transpose(),dif/2)
            #print("gvz: ", gvz.shape)
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
    #cos_Anchor = cosine_similarity(Anchor.transpose(), Anchor.transpose())
    #matlab supports indexing element(i,j) in nxm matrix as i*m+j, python doesn't
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
    #Z_r = np.matmul(Z, np.diag(np.sqrt(np.sum(Z,0)**-1)))
    #Z_l = np.matmul(np.diag(np.sqrt(np.sum(Z,0)**-1)), Z.T)
    #temp_Z1 = np.linalg.multi_dot([np.diag(np.sum(zW,0)**-1), zW.T])
    #temp_Z2 = np.matmul(temp_Z1, Z)
    #W_anchor = np.matmul(Z_l, Z_r)
    W_anchor = np.matmul(Z.T, Z)
    W_diag = np.diag(np.sqrt(np.sum(W_anchor,0)**-1))
    W_anchor = np.linalg.multi_dot([W_diag, W_anchor, W_diag])
    #W_anchor = norm(W_anchor)
    ZW = np.matmul(Z, W_anchor)
    #WZ = np.linalg.multi_dot([zW, temp_Z2])
    #T = np.dot(zW.transpose(), zW)
    #rL = np.linalg.multi_dot([np.diag(np.sqrt(np.sum(T,0))), T, np.diag(np.sqrt(np.sum(T,0))**-1)])
    #WZ = np.linalg.multi_dot([zW, rL])
    #anotherY = np.matmul(anotherY, np.diag(np.sqrt(np.sum(anotherY,0)**-1)))
    return ZW

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
#       m                                      -- num of anchors
#       p                                      -- percentage of NNs to consider
#       cn                                     -- rounds of optimization
# Out:  ZW              (a+...+z, m)           -- OCAT feature matrix
###################################################################
def sparse_encoding_integration_(data_list, m=None, p=0.3, cn=5):
    if m==None:
        m = m_estimate(data_list)
    # find anchors
    anchor_list = find_anchors(data_list, m)
    # construct sparse anchor graph
    s = round(p*m)
    Z_list = []
    for i, dataset in enumerate(data_list):
        dataset_Z_list = []
        for j, anchor in enumerate(anchor_list):
            Z = AnchorGraph(dataset.transpose(), anchor.transpose(), s, 2, cn)
            dataset_Z_list.append(Z)
        dataset_Z = np.concatenate(dataset_Z_list, axis=1)
        Z_list.append(dataset_Z)
    Z = np.nan_to_num(np.concatenate(Z_list, axis=0))
    ZW = Z_to_ZW(Z)
    ZW = norm(np.nan_to_num(ZW))
    return ZW

def sparse_encoding_integration(data_list, m_list=None, s_list=None, p=0.3, cn=5):
    if m_list==None:
        m_list = [m_estimate(d) for d in data_list]
    # find anchors
    anchor_list = find_anchors(data_list, m_list)
    # construct sparse anchor graph
    if s_list ==None:
        s_list = [round(p*m) for m in m_list]
    Z_list = []
    for i, dataset in enumerate(data_list):
        dataset_Z_list = []
        for j, anchor in enumerate(anchor_list):
            Z = AnchorGraph(dataset.transpose(), anchor.transpose(), s_list[j], 2, cn)
            dataset_Z_list.append(Z)
        dataset_Z = np.concatenate(dataset_Z_list, axis=1)
        Z_list.append(dataset_Z)
    Z = np.nan_to_num(np.concatenate(Z_list, axis=0))
    ZW = Z_to_ZW(Z)
    ZW = norm(np.nan_to_num(ZW))
    return ZW


def post_processing_pca(Z, topk=20):
    # center data by standard scaling
    scaler=StandardScaler()
    scaler.fit(Z)
    Z = scaler.transform(Z)
    # take topk PCs
    pca = PCA(n_components=topk, svd_solver='arpack')
    pca_result = pca.fit_transform(Z)
    return pca_result
