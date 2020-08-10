import numpy as np
cimport numpy as np
DTYPE = np.float
DTYPE_int = np.int

cdef SimplexPr(np.ndarray X):
    cdef int C = X.shape[0]
    cdef int N = X.shape[1]
    #to sort in descending order
    X = np.array(X).astype(DTYPE)
    cdef np.ndarray T = np.sort(X,axis=0) [::-1]
    cdef np.ndarray t = np.zeros([C,1], dtype=DTYPE) 
    cdef int i, j, kk, k
    cdef float theta, tep, sum_t
    for i in range(N):
        sum_t = 0
        kk = -1
        t = T[:,i]
        #find kk where first kk element >= 1 otherwise kk = last index
        for j in range(C):
            tep = 0
            for k in range(j+1):
                tep = tep + t[k]
            tep = t[j] - (tep-1)/(j+1)
            #tep = t[j] - (np.sum(t[0:j+1]) - 1)/(j+1)
            if tep <= 0:
                kk = j-1
                break
        if kk == -1:
            kk = C-1
        #scale X to be at 1
        for k in range(kk+1):
            sum_t = sum_t+t[k]
        theta = (sum_t-1)/(kk+1)
        #theta = (np.sum(t[0:kk+1]) - 1)/(kk+1)
        X[:,i] = (X[:,i] - theta).clip(min=0).flatten()
    return X

def LAE (np.ndarray x, np.ndarray U, int cn):
    cdef int d = U.shape[0]
    cdef int s = U.shape[1]
    cdef np.ndarray z0 = np.ones((s,1), dtype=DTYPE)/s
    cdef np.ndarray z1 = z0
    cdef np.ndarray delta = np.zeros((1,cn+2), dtype=DTYPE)
    delta[0][0] = 0
    delta[0][1] = 1
    cdef np.ndarray beta = np.zeros((1,cn+1), dtype=DTYPE)
    beta[0][0] = 1
    cdef int t, j
    cdef float alpha, b
    cdef np.ndarray v = np.zeros([s,1], dtype=DTYPE)
    cdef np.ndarray dif = np.zeros([d,1], dtype=DTYPE)
    cdef np.ndarray gv = np.zeros([1,1], dtype=DTYPE)
    cdef np.ndarray dgv = np.zeros([s,1], dtype=DTYPE)
    cdef np.ndarray z = np.zeros([s,1], dtype=DTYPE)
    cdef np.ndarray gz = np.zeros([1,1], dtype=DTYPE)
    cdef np.ndarray gvz = np.zeros([1,1], dtype=DTYPE)
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
        delta[0][t+2] = (1+np.sqrt(1+4*delta[0][t+1]**2))/2
        if np.sum(abs(z1-z0)) <= 1e-4:
            break
    z = z1
    return z, np.sum(abs(z1-z0))

def find_regression(np.ndarray TrainData, np.ndarray val, np.ndarray pos, Anchor, int s, int cn):
    cdef int i, D, n
    cdef float diff
    D = TrainData.shape[0]
    n = TrainData.shape[1]
    cdef np.ndarray x = np.zeros([D], dtype=DTYPE)
    cdef np.ndarray U = np.zeros([D,s], dtype=DTYPE)
    for i in range(n):
        x = TrainData[:,i]
        x = (x/np.linalg.norm(x,2)).reshape((len(x),1))
        U = Anchor[:,pos[i,:]]
        a = LAE(x,U,cn)
        val[i,:] = a[0].flatten()
        diff = diff + a[1]
    return val, diff
