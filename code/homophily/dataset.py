from org.gesis.model.DPAH import DPAH
import networkx as nx
import numpy as np
import dgl
import torch
import random

def get_graph(graph, feature_dim=5, proxy_correlation=0.9, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, args=None):
    np.random.seed(42)
    random.seed(42)

    total_degree = sum(dict(graph.degree()).values())
    avg_degree = total_degree / graph.number_of_nodes()

    total_nodes = graph.number_of_nodes()
    injected_nodes = int(total_nodes * 0.01)

    avg_degree = round(avg_degree)
    injected_nodes = round(injected_nodes)

    dgl_graph = dgl.from_networkx(graph)

    num_nodes = dgl_graph.num_nodes()
    features = []
    proxy_labels = []
    sensitive_attributes = []
    additional_features = []
    
    for node in range(num_nodes):
        label = graph.nodes[node]['m']
        binary_feature = np.array([label], dtype=np.float32)
        features.append(binary_feature)
        
        sensitive_attributes.append(label)
        
        if label == 0:
            proxy_label = np.random.normal(loc=0.0, scale=1.0) 
        else:
            proxy_label = np.random.normal(loc=1.0, scale=1.0)  
        
        proxy_binary_label = 1 if proxy_label > np.random.uniform(0, 1) * proxy_correlation else 0
        proxy_labels.append(proxy_binary_label)
        
        if proxy_binary_label == 0:
            correlated_feature = np.random.normal(loc=0.5, scale=0.1, size=feature_dim)
        else:
            correlated_feature = np.random.normal(loc=1.0, scale=0.1, size=feature_dim)
        
        additional_features.append(correlated_feature)
    
    features = np.hstack([np.array(features), np.array(additional_features)])
    
    features = torch.tensor(features, dtype=torch.float32)
    proxy_labels = torch.tensor(proxy_labels, dtype=torch.int64)
    sensitive_attributes = torch.tensor(sensitive_attributes, dtype=torch.int64)

    dgl_graph.ndata['feature'] = features
    dgl_graph.ndata['label'] = proxy_labels
    dgl_graph.ndata['sensitive'] = sensitive_attributes
    
    indices = list(range(num_nodes))
    random.shuffle(indices)
    train_split = int(train_ratio * num_nodes)
    val_split = int((train_ratio + val_ratio) * num_nodes)
    
    train_index = torch.tensor(indices[:train_split], dtype=torch.int64)
    val_index = torch.tensor(indices[train_split:val_split], dtype=torch.int64)
    test_index = torch.tensor(indices[val_split:], dtype=torch.int64)
    
    dgl_graph.ndata['train_index'] = torch.zeros(num_nodes, dtype=torch.bool)
    dgl_graph.ndata['val_index'] = torch.zeros(num_nodes, dtype=torch.bool)
    dgl_graph.ndata['test_index'] = torch.zeros(num_nodes, dtype=torch.bool)
    
    dgl_graph.ndata['train_index'][train_index] = True
    dgl_graph.ndata['val_index'][val_index] = True
    dgl_graph.ndata['test_index'][test_index] = True

    dgl_graph = dgl.add_self_loop(dgl_graph)

    index_split = {
        'train_index': train_index,
        'val_index': val_index,
        'test_index': test_index
    }
    
    return dgl_graph, index_split, avg_degree, injected_nodes

