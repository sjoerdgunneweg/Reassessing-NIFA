import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from model import *
from utils import fair_matrix
import random
import itertools

class Bayesian_Network(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, T, theta, device):
        super(Bayesian_Network, self).__init__()
        self.layer1 = nn.Parameter(torch.Tensor(in_dim, hid_dim))
        self.layer2 = nn.Parameter(torch.Tensor(hid_dim, out_dim))
        self.relu = nn.ReLU()
        self.device = device

        self.T = T
        self.theta = theta
        self.mask1 = (torch.rand(T, in_dim, hid_dim) <= theta).to(device)
        self.mask2 = (torch.rand(T, hid_dim, out_dim) <= theta).to(device)

        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.layer1, gain=gain)
        nn.init.xavier_uniform_(self.layer2, gain=gain)

    def forward(self, g):
        result = []
        for t in range(self.T):
            with g.local_scope():
                g.update_all(fn.u_mul_e('feature', 'edge_weight', 'm'), fn.sum('m', 'h'))
                w1 = torch.mul(self.layer1, self.mask1[t])
                g.ndata['h'] = self.relu(torch.mm(g.ndata['h'], w1))

                g.update_all(fn.u_mul_e('h', 'edge_weight', 'm'), fn.sum('m', 'o'))
                w2 = torch.mul(self.layer2, self.mask2[t])
                y = torch.mm(g.ndata['o'], w2)

                result.append(y)
        return result

    def optimize(self, g, lr):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        label = g.ndata['label']
        label_idx = (label >= 0)

        degree = g.in_degrees()
        degree = torch.pow(degree, -0.5)
        src, dst = g.edges()
        g.edata['edge_weight'] = degree[src] * degree[dst]

        for epoch in range(500):
            output = self(g)
            loss = 0
            for y_hat in output:
                loss += loss_fn(y_hat[label_idx], label[label_idx])
            loss = loss / self.T + (1-self.theta) / self.T * ((self.layer1 ** 2).mean() + (self.layer2 ** 2).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.eval()
        with torch.no_grad():
            output = self(g)
            output = torch.stack(output, dim=0)
            output = F.softmax(output, dim=2)
            uncertainty = torch.var(output, dim=0).mean(dim=1)
            return uncertainty


#-----------------------------------our_contribution--------------------------------
class Edge_Attack_Multiclass():
    """
    Part of the attack class that injects edges into the graph which is adapted to handle multi-class sensitive attributes.
    """    
    def __init__(self, node, edge, mode):
        self.node = node
        self.edge = edge - edge % 2
        self.mode = mode

    def attack(self, g, uncertainty, ratio):
        sens = g.ndata['sensitive']
        sens_unique_vals = torch.unique(sens)
        sens_unique_vals = sens_unique_vals[sens_unique_vals != -1]

        masks = []
        for sensitive_attr in sens_unique_vals:
            mask = torch.logical_and(g.ndata['sensitive']==sensitive_attr, g.ndata['label']>=0)
            masks.append(mask)

        if self.mode == "uncertainty":
            idxs = []
            for i,sensitive_attr in enumerate(sens_unique_vals):
                unc = torch.where(masks[i], uncertainty, 0)
                _, idx = torch.sort(unc, descending=True)
                idxs.append(idx)
        elif self.mode == "degree":
            idxs = []
            for i,sensitive_attr in enumerate(sens_unique_vals):
                degree = g.in_degrees() + g.out_degrees()
                unc = torch.where(masks[i], degree, 1000)
                _, idx = torch.sort(unc, descending=True)
                idxs.append(idx)
        else:
            raise NotImplementedError


        for i in range(len(idxs)):
            idxs[i] = idxs[i][:int(ratio*sum(masks[i]))]

        # inject nodes
        g.add_nodes(self.node)
        g.ndata['label'][-self.node:] = -1
        g.ndata['sensitive'][-self.node:] = -1

        # inject edges
        N = g.num_nodes()

        # assigning the number of nodes to be injected to each sensitive group by balancing the number of nodes to be injected to exhuastion:
        node_budgets = []
        base, remainder = divmod(self.node, len(idxs))
        for i in range(len(idxs)):
            node_budgets.append(base + (1 if i < remainder else 0))

        src = []
        for i,idx in enumerate(idxs):
            src+=[idx[random.sample(range(len(idx)), self.edge)] for _ in range(node_budgets[i])]
        src = torch.cat(src, dim=0)

        dst = torch.t(torch.arange(N-self.node, N).repeat(self.edge, 1)).flatten()

        src = src.to(g.device)
        dst = dst.to(g.device)
        
        g.add_edges(src, dst)
        g.add_edges(dst, src)

        return g
#-----------------------------------------------------------------------------------    
    

class Edge_Attack():
    def __init__(self, node, edge, mode):
        self.node = node
        self.edge = edge - edge % 2
        self.mode = mode

    def attack(self, g, uncertainty, ratio):
        mask_a = torch.logical_and(g.ndata['sensitive']==1, g.ndata['label']>=0)
        mask_d = torch.logical_and(g.ndata['sensitive']==0, g.ndata['label']>=0)
        if self.mode == "uncertainty":
            unc_a = torch.where(mask_a, uncertainty, 0)
            _, idx_a = torch.sort(unc_a, descending=True)
            unc_d = torch.where(mask_d, uncertainty, 0)
            _, idx_d = torch.sort(unc_d, descending=True)
        elif self.mode == "degree":
            degree = g.in_degrees() + g.out_degrees()
            unc_a = torch.where(mask_a, degree, 1000)
            _, idx_a = torch.sort(unc_a, descending=False)
            unc_d = torch.where(mask_d, degree, 1000)
            _, idx_d = torch.sort(unc_d, descending=False)
        else:
            raise NotImplementedError

        idx_a = idx_a[:int(ratio*sum(mask_a))]
        idx_d = idx_d[:int(ratio*sum(mask_d))]

        # inject nodes
        g.add_nodes(self.node)
        g.ndata['label'][-self.node:] = -1
        g.ndata['sensitive'][-self.node:] = -1

        # inject edges
        N = g.num_nodes()

        node_budget_a = self.node // 2
        node_budget_d = self.node - node_budget_a

        src = torch.cat([idx_a[random.sample(range(len(idx_a)), self.edge)] for _ in range(node_budget_a)] + 
                        [idx_d[random.sample(range(len(idx_d)), self.edge)] for _ in range(node_budget_d)], dim=0)
        
        # src2 = torch.cat([idx_a[torch.randint(0, len(idx_a), (self.edge,), device=g.device)] for _ in range(node_budget_a)] + 
        #                 [idx_d[torch.randint(0, len(idx_d), (self.edge,), device=g.device)] for _ in range(node_budget_d)], dim=0)
        
        # print(src)

        # print(src2)

        dst = torch.t(torch.arange(N-self.node, N).repeat(self.edge, 1)).flatten()

        src = src.to(g.device)
        dst = dst.to(g.device)
        
        g.add_edges(src, dst)
        g.add_edges(dst, src)

        return g


#-----------------------------------our_contribution--------------------------------
class Feature_Attack_Multiclass(nn.Module):   
    """
    Part of the attack class that optimizes the features of the injected nodes which is adapted to handle multi-class sensitive attributes.
    """     
    def __init__(self, g, in_dim, hid_dim, out_dim, node, sensitive_attr_mode):
        super(Feature_Attack_Multiclass, self).__init__()
        self.model = GCN(in_dim, hid_dim, out_dim)

        self.mode = sensitive_attr_mode
        feature = g.ndata['feature']
        self.lower_bound = torch.min(feature, dim=0)[0].repeat(node, 1)
        self.upper_bound = torch.max(feature, dim=0)[0].repeat(node, 1)
        self.feature = nn.Parameter(torch.zeros(node, in_dim).normal_(mean=0.5,std=0.5))

        self.node = node

    def forward(self, g):
        return self.model(g, torch.cat((g.ndata["feature"][:-self.node], self.feature), dim=0))

    def optimize(self, g, index_split, epochs, lr, alpha, beta, loops=50):
        train_index = index_split['train_index']
        val_index = index_split['val_index']
        test_index = index_split['test_index']
        label = g.ndata['label']
        sensitive = g.ndata['sensitive']
        sens_unique_vals = torch.unique(sensitive)
        sens_unique_vals = sens_unique_vals[sens_unique_vals != -1]

        label_train = g.ndata['label'][train_index]

        idxs = []
        for i,sensitive_attr in enumerate(sens_unique_vals):
            idx = (sensitive[train_index] == sensitive_attr)
            idxs.append(idx)   
        
        optimizer_W = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0)
        optimizer_F = torch.optim.Adam([self.feature], lr=lr, weight_decay=0)
        cross_entropy = nn.CrossEntropyLoss()

        for rounds in range(0, epochs, loops):
            for epoch in range(rounds, rounds+loops):
                output = self(g)
                loss_W = cross_entropy(output[train_index], label[train_index])
                optimizer_W.zero_grad()
                loss_W.backward()
                optimizer_W.step()
                
            for epoch in range(rounds, rounds+loops):
                output = self(g)
                output_train = output[train_index]
                
                if self.mode == 'OvA':
                    # loss_sp
                    losses_sp = []
                    # looping over all possible sensitive attributes and compute a mse loss for each pair of sensitive group and remainder:
                    for idx in idxs:
                        out_sensitive_group = torch.mean(output_train[idx], dim=0).to(g.device)
                        out_remainder = torch.mean(output_train[idx == 0], dim=0).to(g.device)
                        loss_sp_case = beta * F.mse_loss(out_sensitive_group, out_remainder)
                        losses_sp.append(loss_sp_case)
                    
                    loss_sp = torch.mean(torch.stack(losses_sp))

                    # loss_eo
                    losses_eo = []
                    # looping over all possible sensitive attributes and compute a mse loss for each pair of sensitive group and remainder:
                    for idx in idxs:
                        embed_sensitive_group = torch.zeros(label_train.max()+1).to(g.device)
                        embed_remainder = torch.zeros(label_train.max()+1).to(g.device)

                        for i in range(label_train.max()+1):
                            idx_sensitive_group_c = torch.where(label_train[idx] == i)[0]
                            idx_remainder_c = torch.where(label_train[idx == 0] == i)[0]
                            mean_sensitive_group = torch.mean(output_train[idx][idx_sensitive_group_c], dim=0)
                            mean_remainder = torch.mean(output_train[idx == 0][idx_remainder_c], dim=0)
                            embed_sensitive_group[i] = mean_sensitive_group[i]
                            embed_remainder[i] = mean_remainder[i]
                        loss_eo_case = beta * F.mse_loss(embed_sensitive_group, embed_remainder)
                        losses_eo.append(loss_eo_case)
                    
                    loss_eo = torch.mean(torch.stack(losses_eo))

                elif self.mode == 'OvO':
                    # loss_sp
                    losses_sp = []
                    # looping over all possible pairs of sensitive attributes and compute a mse loss for each pair of sensitive group:
                    for sens_a, sens_b in itertools.combinations(range(len(idxs)), 2):
                        out_a = torch.mean(output_train[idxs[sens_a]], dim=0).to(g.device)
                        out_b = torch.mean(output_train[idxs[sens_b]], dim=0).to(g.device)
                        loss_sp_case = beta * F.mse_loss(out_a, out_b)
                        losses_sp.append(loss_sp_case)
                    loss_sp = torch.mean(torch.stack(losses_sp))

                    # loss_eo
                    losses_eo = []
                    # looping over all possible pairs of sensitive attributes and compute a mse loss for each pair of sensitive group:
                    for sens_a, sens_b in itertools.combinations(range(len(idxs)),2):
                        embed_a = torch.zeros(label_train.max()+1).to(g.device)
                        embed_b = torch.zeros(label_train.max()+1).to(g.device)

                        for i in range(label_train.max()+1):
                            idx_a_c = torch.where(label_train[idxs[sens_a]] == i)[0]
                            idx_b_c = torch.where(label_train[idxs[sens_b]] == i)[0]
                            mean_a = torch.mean(output_train[idxs[sens_a]][idx_a_c], dim=0)
                            mean_b = torch.mean(output_train[idxs[sens_b]][idx_b_c], dim=0)
                            embed_a[i] = mean_a[i]
                            embed_b[i] = mean_b[i]
                        loss_eo_case = beta * F.mse_loss(embed_a, embed_b)
                        losses_eo.append(loss_eo_case)
                    loss_eo = torch.mean(torch.stack(losses_eo))
                    
                else:
                    raise NotImplementedError("The sensitive_attr_mode does not work with the current implementation of the feature attack.")
                
                # loss_cf
                loss_cf = alpha * F.mse_loss((self.feature[:self.node//2]).mean(dim=0), (self.feature[self.node//2:]).mean(dim=0))

                # loss_ce
                loss_ce = cross_entropy(output_train, label_train)

                loss_F = -loss_sp - loss_eo - loss_cf + loss_ce
                optimizer_F.zero_grad()
                loss_F.backward()
                optimizer_F.step()
                self.feature.data = torch.clamp(self.feature.data, self.lower_bound, self.upper_bound) 
            
        g.ndata['feature'][-self.node:] = torch.round(self.feature).detach()
        return g
#-----------------------------------------------------------------------------------    



class Feature_Attack(nn.Module):    
    def __init__(self, g, in_dim, hid_dim, out_dim, node):
        super(Feature_Attack, self).__init__()
        self.model = GCN(in_dim, hid_dim, out_dim)

        feature = g.ndata['feature']
        self.lower_bound = torch.min(feature, dim=0)[0].repeat(node, 1)
        self.upper_bound = torch.max(feature, dim=0)[0].repeat(node, 1)
        self.feature = nn.Parameter(torch.zeros(node, in_dim).normal_(mean=0.5,std=0.5))

        self.node = node

    def forward(self, g):
        return self.model(g, torch.cat((g.ndata["feature"][:-self.node], self.feature), dim=0))

    def optimize(self, g, index_split, epochs, lr, alpha, beta, loops=50):
        train_index = index_split['train_index']
        val_index = index_split['val_index']
        test_index = index_split['test_index']
        label = g.ndata['label']
        sensitive = g.ndata['sensitive']

        label_train = g.ndata['label'][train_index]  
        idx_a = torch.where(sensitive[train_index] == 1)[0]  
        idx_d = torch.where(sensitive[train_index] == 0)[0]
        
        optimizer_W = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0)
        optimizer_F = torch.optim.Adam([self.feature], lr=lr, weight_decay=0)
        cross_entropy = nn.CrossEntropyLoss()

        for rounds in range(0, epochs, loops):
            for epoch in range(rounds, rounds+loops):
                output = self(g)
                loss_W = cross_entropy(output[train_index], label[train_index])
                optimizer_W.zero_grad()
                loss_W.backward()
                optimizer_W.step()
                
            for epoch in range(rounds, rounds+loops):
                output = self(g)
                output_train = output[train_index]

                # loss_sp
                a_sp_embed = torch.mean(output_train[idx_a], dim=0).to(g.device)
                d_sp_embed = torch.mean(output_train[idx_d], dim=0).to(g.device)
                loss_sp = beta * F.mse_loss(a_sp_embed, d_sp_embed)

                # loss_eo
                a_eo_embed = torch.zeros(label_train.max()+1).to(g.device)
                d_eo_embed = torch.zeros(label_train.max()+1).to(g.device)
                for i in range(label_train.max()+1):
                    idx_ac = torch.where(label_train[idx_a] == i)[0]
                    idx_dc = torch.where(label_train[idx_d] == i)[0]
                    mean_a = torch.mean(output_train[idx_a][idx_ac], dim=0)
                    mean_d = torch.mean(output_train[idx_d][idx_dc], dim=0)
                    a_eo_embed[i] = mean_a[i]
                    d_eo_embed[i] = mean_d[i]
                loss_eo = beta * F.mse_loss(a_eo_embed, d_eo_embed)

                # loss_cf
                loss_cf = alpha * F.mse_loss((self.feature[:self.node//2]).mean(dim=0), (self.feature[self.node//2:]).mean(dim=0))

                # loss_ce
                loss_ce = cross_entropy(output_train, label_train)

                loss_F = -loss_sp - loss_eo - loss_cf + loss_ce
                optimizer_F.zero_grad()
                loss_F.backward()
                optimizer_F.step()
                self.feature.data = torch.clamp(self.feature.data, self.lower_bound, self.upper_bound) 
            
        g.ndata['feature'][-self.node:] = torch.round(self.feature).detach()
        return g


class Attacker():
    def __init__(self, g, in_dim, hid_dim, out_dim, device, args):
        self.args = args
        self.bayesian_network = Bayesian_Network(in_dim, hid_dim, out_dim, args.T, args.theta, device).to(device)
        # Check whether the sensitive attribute mode is binary or multi-class:
        #-----------------------------------our_contribution--------------------------------
        if args.sensitive_attr_mode != 'Binary':
            self.edge_attack = Edge_Attack_Multiclass(args.node, args.edge, args.mode)
            self.feature_attack = Feature_Attack_Multiclass(g, in_dim, hid_dim, out_dim, args.node, args.sensitive_attr_mode).to(device)

        else:
            self.edge_attack = Edge_Attack(args.node, args.edge, args.mode)
            self.feature_attack = Feature_Attack(g, in_dim, hid_dim, out_dim, args.node).to(device)
        #-----------------------------------------------------------------------------------
    def attack(self, g, index_split):
        uncertainty = self.bayesian_network.optimize(g, self.args.lr)
        g = self.edge_attack.attack(g, uncertainty, self.args.ratio)
        g = self.feature_attack.optimize(g, index_split, self.args.epochs, self.args.lr, self.args.alpha, self.args.beta, self.args.loops)
        return g, uncertainty

