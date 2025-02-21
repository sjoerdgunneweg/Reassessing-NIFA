import pickle
from dgl import graph
import scipy.sparse as sparse
import numpy as np
import torch
import dgl
import random

#-----------------------------------our_contribution--------------------------------
def set_random_seeds(seed=42):
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    dgl.random.seed(seed)  

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#-----------------------------------------------------------------------------------

#-----------------------------------our_contribution--------------------------------
def combine_features(adj,features,add_adj,add_features):

    nfeature=np.concatenate([features,add_features],0)
    
    total = len(features)
    adj_added1 = add_adj[:,:total]
    adj = sparse.vstack([adj,adj_added1])
    
    adj = sparse.hstack([adj,add_adj.transpose()])
    for i in range(len(adj.data)):
        if (adj.data[i]!=0) and (adj.data[i]!=1):
            adj.data[i] = 1

    return adj.tocsr(),nfeature
#-----------------------------------------------------------------------------------

#-----------------------------------our_contribution--------------------------------
def load_bin(dataset):
    glist, _ = dgl.load_graphs(f"../data/{dataset}.bin")
    g = glist[0]

    src, dst = g.edges()
    num_nodes = g.num_nodes()

    adj = sparse.coo_matrix((np.ones_like(src.numpy()), (src.numpy(), dst.numpy())),
                        shape=(num_nodes, num_nodes)).tocsr()

    features = g.ndata["feature"].numpy()
    labels = g.ndata["label"].numpy()
    sensitive = g.ndata["sensitive"].numpy()

    return adj, features, labels, sensitive
#-----------------------------------------------------------------------------------

#-----------------------------------our_contribution--------------------------------
def load_injected_data(dataset):
    with open (f'../TDGIA/tdgia_nodes/{dataset}_gcn_nifa/adj.pkl', 'rb') as f:
        inc_adj = pickle.load(f)

    inc_feat = np.load(f'../TDGIA/tdgia_nodes/{dataset}_gcn_nifa/feature.npy')

    return inc_adj, inc_feat

def to_dgl_graph(adj, features):
    src, dst = adj.nonzero()
    g = graph((src, dst))

    g.ndata['feature'] = torch.tensor(features, dtype=torch.float32)

    return g
#-----------------------------------------------------------------------------------