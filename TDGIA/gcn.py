import scipy.sparse as sp
import numpy as np

def GCNadj(adj,pow=-0.5):
    adj2=sp.eye(adj.shape[0])+adj
    for i in range(len(adj2.data)):
        if (adj2.data[i]>0 and adj2.data[i]!=1):
            adj2.data[i]=1
    adj2 = sp.coo_matrix(adj2)
    
    rowsum = np.array(adj2.sum(1))
    d_inv_sqrt = np.power(rowsum, pow).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj2=d_mat_inv_sqrt @ adj2 @ d_mat_inv_sqrt
    
    return adj2.tocoo()