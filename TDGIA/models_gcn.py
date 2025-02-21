# import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import dgl 

    
def SAGEadj(adj,pow=-1):
    adj2=sp.eye(adj.shape[0])*(1)+adj
    for i in range(len(adj2.data)):
        if (adj2.data[i]>0 and adj2.data[i]!=1):
            adj2.data[i]=1
        if (adj2.data[i]<0):
            adj2.data[i]=0
    adj2.eliminate_zeros()
    adj2 = sp.coo_matrix(adj2)
    if pow==0:
        return adj2.tocoo()
    rowsum = np.array(adj2.sum(1))
    d_inv_sqrt = np.power(rowsum, pow).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj2=d_mat_inv_sqrt @ adj2
    
    return adj2.tocoo()

# -----------------------------------NIFA's GCN--------------------------------------------

class gcn_nifa(nn.Module):
    def __init__(self, in_feats, num_classes, h_feats=128):
        super(gcn_nifa, self).__init__()

        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats, norm='both')
        self.conv2 = dgl.nn.GraphConv(h_feats, num_classes, norm='both')

    def forward(self, g, in_feat, dropout=0):

        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

#------------------------------------------------------------------------------------------
