# %%
import numpy as np
import scipy.sparse as sp
import torch
import os
import pandas as pd
import dgl
import random
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score, roc_curve
from scipy.spatial import distance_matrix
import networkx as nx
from scipy.sparse.csgraph import connected_components

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, desc=None):
        return x

def fair_metric(labels, output, idx, sens, status):
    mid_result = {}
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

    idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)
    idx_s0_y0 = np.bitwise_and(idx_s0, val_y == 0)
    idx_s1_y0 = np.bitwise_and(idx_s1, val_y == 0)

    pred_y = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0]) / sum(idx_s0) -
                 sum(pred_y[idx_s1]) / sum(idx_s1))
    mid_result['yp1.a1'] = sum(pred_y[idx_s1]) / sum(idx_s1)
    mid_result['yp1.a0'] = sum(pred_y[idx_s0]) / sum(idx_s0)
    equality = abs(sum(pred_y[idx_s0_y1]) /
                   sum(idx_s0_y1) -
                   sum(pred_y[idx_s1_y1]) /
                   sum(idx_s1_y1))
    mid_result['yp1.y1a1'] = sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1)
    mid_result['yp1.y1a0'] = sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1)
    eq_odds = abs(sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1)) + \
              abs(sum(pred_y[idx_s0_y0]) / sum(idx_s0_y0) - sum(pred_y[idx_s1_y0]) / sum(idx_s1_y0))
    mid_result['yp1.y0a1'] = sum(pred_y[idx_s1_y0]) / sum(idx_s1_y0)
    mid_result['yp1.y0a0'] = sum(pred_y[idx_s0_y0]) / sum(idx_s0_y0)
    # dis_imp = (sum(pred_y[idx_s1]) / sum(idx_s1)) / (sum(pred_y[idx_s0]) / sum(idx_s0))
    mid_result['y1a1'] = sum(idx_s1_y1)
    mid_result['y1a0'] = sum(idx_s0_y1)
    mid_result['y0a1'] = sum(idx_s1_y0)
    mid_result['y0a0'] = sum(idx_s0_y0)
    # print(status,':y1a1:',mid_result['y1a1']) # evaluate the label and SA distribution
    # print(status,':y1a0:',mid_result['y1a0'])
    # print(status,':y0a1:',mid_result['y0a1'])
    # print(status,':y0a0:',mid_result['y0a0'])
    # print(status, ':y1:', mid_result['y1a1']+mid_result['y1a0'])
    # print(status, ':y0:', mid_result['y0a1']+mid_result['y0a0'])
    return parity, equality, eq_odds, mid_result  # ,dis_imp,mid_result

def check_dataset(dataset,adj,labels,sens,idx_train,idx_val,idx_test):
    # if dataset not in ['nba','region_job','region_job_2']:
    adj=adj.tocoo()
    row,col=adj.row,adj.col
    print("num edges:",len(row)//2)
    labels = labels.cpu()
    label_idx = np.where(labels >= 0)[0]

    print("num labels:", len(label_idx))
    # check label balancing
    label_idx_0 = set(np.where(labels == 0)[0])
    label_idx_1 = set(np.where(labels == 1)[0])
    print("num labels 0:", len(label_idx_0))
    # print("num labels 1:", len(label_idx) - len(label_idx_0))
    print("num labels 1:", len(label_idx_1))


    sens_idx = set(np.where(sens >= 0)[0])
    print("num nodes with sa:", len(sens_idx))

    idx_label_sens=set(label_idx)&sens_idx
    print("num nodes with label and sa:", len(idx_label_sens))
    # check sa
    sens_idx_0 = set(np.where(sens == 0)[0])
    sens_idx_1 = set(np.where(sens == 1)[0])
    print("num nodes with sa=0:", len(sens_idx_0))
    print("num nodes with sa=1:", len(sens_idx_1))


    idx_y1s1=sens_idx_1&label_idx_1
    idx_y1s0=sens_idx_0&label_idx_1
    idx_y0s1=sens_idx_1&label_idx_0
    idx_y0s0=sens_idx_0&label_idx_0
    print("y1s1:",len(idx_y1s1)/len(idx_label_sens))
    print("y1s0:",len(idx_y1s0)/len(idx_label_sens))
    print("y0s1:",len(idx_y0s1)/len(idx_label_sens))
    print("y0s0:",len(idx_y0s0)/len(idx_label_sens))

    # save edge rate file-------------------------------------
    # homo_edges_rate = get_density_matrix(adj,labels,sens)
    #
    # fname = dataset + "_homo_edges_rate.csv"
    # np.savetxt(fname, homo_edges_rate, delimiter=",")
    #---------------------------------------------------------

    # fname=dataset+"_homo_edges.csv"
    # np.savetxt(fname,homo_edges,delimiter=",")


    print(f"Data splits: {len(idx_train)} train, {len(idx_val)} val, {len(idx_test)} test. ")

def load_attacked_graph(
        dataset='./NBAattack100edges/nba',
        adj_fname='nba_adj_after_attack.npz'):
    adj = sp.load_npz(adj_fname)  #
    features = sp.load_npz(dataset + '_feature.npz')  #
    labels = np.load(dataset + "_label.npy")
    idx_train = np.load(dataset + "_train_idx.npy")
    idx_val = np.load(dataset + "_val_idx.npy")
    idx_test = np.load(dataset + "_test_idx.npy")
    sens = np.load(dataset + "_sens.npy")

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    sens = torch.FloatTensor(sens)
    return adj, features, labels, idx_train, idx_val, idx_test, sens

def rand_attack(adj, edge_perturbations):
    print("Begin random attack...")
    perturbations = int(edge_perturbations * (adj.sum() // 2))
    _N = adj.shape[0]
    adj2 = adj.todense()
    for _it in tqdm(range(perturbations), desc="Perturbing graph"):
        attack_nodes = random.choices(np.arange(_N), k=2)
        if adj[attack_nodes[0], attack_nodes[1]] == 1.0:
            adj2[attack_nodes[0], attack_nodes[1]] = 0.0
            adj2[attack_nodes[1], attack_nodes[0]] = 0.0
        else:
            adj2[attack_nodes[0], attack_nodes[1]] = 1.0
            adj2[attack_nodes[1], attack_nodes[0]] = 1.0
    adj = sp.csr_matrix(adj2)
    return adj

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]

    return 2 * (features - min_values).div(max_values - min_values) - 1  # -1~1

def classification_metrics(output, labels):
    output = output.squeeze()
    preds = (output > 0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    acc = correct / len(labels)

    roc = roc_auc_score(
        labels.cpu().numpy(),
        output.detach().cpu().numpy())

    p = precision_score(labels.cpu().numpy(),
                        preds.detach().cpu().numpy(),
                        zero_division=1)
    r = recall_score(labels.cpu().numpy(),
                     preds.detach().cpu().numpy())  #
    maf1 = f1_score(
        labels.cpu().numpy(),
        preds.detach().cpu().numpy(),
        average='macro')
    mif1 = f1_score(
        labels.cpu().numpy(),
        preds.detach().cpu().numpy(),
        average='micro')

    return acc, roc, p, r, maf1, mif1


def fair_matrix(pred, label, sens, index):

    SP = []
    EO = []

    idx_d = torch.where(sens[index]==0)[0]
    idx_a = torch.where(sens[index]==1)[0]
    for i in range(label.max()+1):
        # SP
        p_i0 = torch.where(pred[index][idx_d] == i)[0]
        p_i1 = torch.where(pred[index][idx_a] == i)[0]

        sp = (p_i1.shape[0]/idx_a.shape[0]) - (p_i0.shape[0]/idx_d.shape[0])
        SP.append(sp)
        
        # EO
        p_y0 = torch.where(label[index][idx_d] == i)[0]
        p_y1 = torch.where(label[index][idx_a] == i)[0]

        p_iy0 = torch.where(pred[index][idx_d][p_y0] == i)[0]
        p_iy1 = torch.where(pred[index][idx_a][p_y1] == i)[0]

        if p_y0.shape[0] == 0 or p_y1.shape[0] == 0:
            eo = 0
        else:
            eo = (p_iy1.shape[0]/p_y1.shape[0]) - (p_iy0.shape[0]/p_y0.shape[0])
        EO.append(eo)   
    return SP, EO

def load_data_nifa(args):
    glist, _ = dgl.load_graphs(f"../data/{args.dataset}.bin")

    device = torch.device("cuda", 0)
    g = glist[0]

    src, dst = g.edges()
    num_nodes = g.num_nodes()
    adj = sp.coo_matrix((np.ones_like(src.cpu().numpy()), 
                         (src.cpu().numpy(), dst.cpu().numpy())),
                        shape=(num_nodes, num_nodes), dtype=np.float32).tocsr()

    features = g.ndata['feature']
    labels = g.ndata['label']
    sens = g.ndata['sensitive']

    random.seed(0)
    label_idx = np.where(labels.cpu().numpy() >= 0)[0]
    random.shuffle(label_idx)

    train_percent_atk = args.train_percent_atk
    train_percent_gnn = args.train_percent_gnn
    val_percent = 0.25

    idx_train_gnn = label_idx[:int(train_percent_gnn * len(label_idx))]
    idx_train_atk = label_idx[:int(train_percent_atk * len(label_idx))]
    idx_val = label_idx[int(train_percent_gnn * len(label_idx)):int((train_percent_gnn + val_percent) * len(label_idx))]
    idx_test = label_idx[int((train_percent_gnn + val_percent) * len(label_idx)):]

    sens_idx = set(np.where(sens.cpu().numpy() >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))

    if len(idx_test) > len(idx_val):
        idx_test = idx_test[:len(idx_val)]

    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:args.sens_number])

    idx_train_gnn = torch.LongTensor(idx_train_gnn).to(device)
    idx_train_atk = torch.LongTensor(idx_train_atk).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    dataset = args.dataset
    sens_attr = args.sensitive
    sens_number = args.sens_number

    return g, adj, features, labels, idx_train_atk, idx_train_gnn, idx_val, idx_test, sens, idx_sens_train, dataset, sens_attr, sens_number

def calc_rate(labels,args):
    if args.dataset == 'pokec_z':
        injected = 102
    elif args.dataset == 'pokec_n':
        injected = 87
    else:
        injected = 32

    labels = torch.where(labels >= 0)[0]
    total = labels.shape[0]  # Get the total number of nodes (first dimension)
    print(f"Total nodes in {args.dataset}: {total}")

    rate = injected / total  # Calculate perturbation rate
    print(f"Perturbation rate for {args.dataset}: {rate:.4f}")
    
    return rate
