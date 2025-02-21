from org.gesis.model.DPAH import DPAH
import networkx as nx
import numpy as np
import dgl
import torch
import random

def get_graph(graph, feature_dim=3, proxy_correlation=0.8, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, args=None):
    np.random.seed(42)
    random.seed(42)

    total_degree = sum(dict(graph.degree()).values())
    avg_degree = total_degree / graph.number_of_nodes()

    total_nodes = graph.number_of_nodes()
    injected_nodes = int(total_nodes * 0.01)

    avg_degree = round(avg_degree)
    injected_nodes = round(injected_nodes)
    
    # Convert the NetworkX graph to a DGL graph
    dgl_graph = dgl.from_networkx(graph)
    
    num_nodes = dgl_graph.num_nodes()
    features = []
    proxy_labels = []
    sensitive_attributes = []
    additional_features = []
    
    for node in range(num_nodes):
        # Access the label directly
        label = graph.nodes[node]['m']
        binary_feature = np.array([label], dtype=np.float32)
        features.append(binary_feature)
        
        # Store the original label as a sensitive attribute
        sensitive_attributes.append(label)
        
        # Generate proxy label from two normal distributions
        if label == 0:
            proxy_label = np.random.normal(loc=0.0, scale=1.0)  # Majority group distribution
        else:
            proxy_label = np.random.normal(loc=1.0, scale=1.0)  # Minority group distribution
        
        # Binarize proxy label based on correlation threshold
        proxy_binary_label = 1 if proxy_label > np.random.uniform(0, 1) * proxy_correlation else 0
        proxy_labels.append(proxy_binary_label)
        
        # Generate additional features with correlation to the proxy task
        if proxy_binary_label == 0:
            correlated_feature = np.random.normal(loc=0.5, scale=0.1, size=feature_dim)
        else:
            correlated_feature = np.random.normal(loc=1.0, scale=0.1, size=feature_dim)
        
        additional_features.append(correlated_feature)
    
    # Combine binary features and additional features
    features = np.hstack([np.array(features), np.array(additional_features)])
    
    # Convert to torch tensors
    features = torch.tensor(features, dtype=torch.float32)
    proxy_labels = torch.tensor(proxy_labels, dtype=torch.int64)
    sensitive_attributes = torch.tensor(sensitive_attributes, dtype=torch.int64)
    
    # Assign data to DGL graph
    dgl_graph.ndata['feature'] = features
    dgl_graph.ndata['label'] = proxy_labels
    dgl_graph.ndata['sensitive'] = sensitive_attributes
    
    # Create train, validation, and test splits
    indices = list(range(num_nodes))
    random.shuffle(indices)
    train_split = int(train_ratio * num_nodes)
    val_split = int((train_ratio + val_ratio) * num_nodes)
    
    train_index = torch.tensor(indices[:train_split], dtype=torch.int64)
    val_index = torch.tensor(indices[train_split:val_split], dtype=torch.int64)
    test_index = torch.tensor(indices[val_split:], dtype=torch.int64)
    
    # Add split indices as node data
    dgl_graph.ndata['train_index'] = torch.zeros(num_nodes, dtype=torch.bool)
    dgl_graph.ndata['val_index'] = torch.zeros(num_nodes, dtype=torch.bool)
    dgl_graph.ndata['test_index'] = torch.zeros(num_nodes, dtype=torch.bool)
    
    dgl_graph.ndata['train_index'][train_index] = True
    dgl_graph.ndata['val_index'][val_index] = True
    dgl_graph.ndata['test_index'][test_index] = True

    dgl_graph = dgl.add_self_loop(dgl_graph)
    
    # Return the graph and index splits
    index_split = {
        'train_index': train_index,
        'val_index': val_index,
        'test_index': test_index
    }
    
    return dgl_graph, index_split, avg_degree, injected_nodes

# g = DPAH(N=100, fm=0.5, d=0.001, plo_M=2.5, plo_m=2.5, h_MM=0.5, h_mm=0.5, verbose=False, seed=1)
# g, index_split = convert_to_dgl(g, feature_dim=3, proxy_correlation=0.8)

# # Access train, validation, and test indices
# train_index = index_split['train_index']
# val_index = index_split['val_index']
# test_index = index_split['test_index']

# print("Train indices:", train_index)
# print("Validation indices:", val_index)
# print("Test indices:", test_index)

# in_dim = g.ndata['feature'].shape[1]
# hid_dim = 128
# out_dim = max(g.ndata['label']).item() + 1
# label = g.ndata['label']

# print(in_dim)
# print(hid_dim)
# print(out_dim)
# print(label)

