import random
import numpy as np
import torch
import dgl
import scipy.sparse as sp


def set_random_seeds(seed=42):
    """    
    Set random seeds for reproducibility across various libraries.

    Args:
        seed: Defaults to 42.
    """    
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    dgl.random.seed(seed)  

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_bins(dataset, gpu):
    """

    Args:
        dataset: dataset name
        gpu: gpu device number

    Returns:
        g: dgl graph
        adj: adjacency matrix
        features: node features
        labels: node labels
        train_index: training index
        val_index: validation index
        test_index: test index
        trainlabels: training labels
        vallabels: validation labels
        testlabels: test labels
    """    

    # load the dataset from the .bin file
    glist, _ = dgl.load_graphs(f"../data/{dataset}.bin")
    g = glist[0]

    # creates the adjacency matrix from the graph
    src, dst = g.edges()
    num_nodes = g.num_nodes()
    adj = sp.coo_matrix((np.ones_like(src.numpy()), (src.numpy(), dst.numpy())),
                        shape=(num_nodes, num_nodes)).tocsr()
    
    # creates the features and labels from the graph
    features = g.ndata["feature"].numpy()
    labels = g.ndata["label"].numpy()

    # creates the training, validation and test indices from the graph
    train_index = torch.where(g.ndata["train_index"])[0].numpy()
    val_index = torch.where(g.ndata["val_index"])[0].numpy()
    test_index = torch.where(g.ndata["test_index"])[0].numpy()

    # creates the training, validation and test labels from the graph
    trainlabels = labels[train_index]
    vallabels = labels[val_index]
    testlabels = labels[test_index]

    # moves the graph to the GPU
    device = torch.device("cuda", gpu)
    g = g.to(device)

    return g, adj, features, labels, train_index, val_index, test_index, trainlabels, vallabels, testlabels