import torch
import numpy as np
import argparse
import time
import dgl
from extra_utils import *

set_random_seeds()

from model import VictimModel
from attack import *
from utils import load_data

parser = argparse.ArgumentParser(description="Fairness Attack Source code")

parser.add_argument('--dataset', default='pokec_z', choices=['pokec_z','pokec_n','dblp'])

parser.add_argument('--hid_dim', type=int, default=128, help='hidden dimension')
parser.add_argument('--T', type=int, default=20, help='sampling times of Bayesian Network')
parser.add_argument('--theta', type=float, default=0.5, help='bernoulli parameter of Bayesian Network')
parser.add_argument('--node', type=int, default=102, help='budget of injected nodes')
parser.add_argument('--edge', type=int, default=50, help='budget of degrees')
parser.add_argument('--alpha', type=float, default=1, help='weight of loss_cf')
parser.add_argument('--beta', type=float, default=1, help='weight of loss_fair')
parser.add_argument('--defense', type=float, default=0, help='the ratio of defense')

parser.add_argument('--ratio', type=float, default=0.5, help='node of top ratio uncertainty are attacked')
parser.add_argument('--before', action='store_true')
parser.add_argument('--models', type=str, nargs="*", default=[])
parser.add_argument('--loops', type=int, default=50)

parser.add_argument('--mode', type=str, default="uncertainty", choices=['uncertainty','degree'], help='principle for selecting target nodes')

parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--patience', type=int, default=50, help='early stop patience')
parser.add_argument('--n_times', type=int, default=1, help='times to run')

#-----------------------For multi-class sensitive attribute NIFA-------------------
# Added for handling multi-class sensitive attributes:
parser.add_argument('--sensitive_attr_mode', type=str, default='Binary', choices=['Binary','OvA', 'OvO'], help='sensitive attribute mode: Binary sensitive attributes or multiclass One vs All sensitive attributes')
# Added for handling the multi-class binning of pokec:
parser.add_argument('--pokec_age_bin', action='store_true', help='bin the age feature in pokec dataset')
parser.add_argument('--device', type=int, default=0, help='device ID for GPU')
#----------------------------------------------------------------------------------


#-----------------------------------our_contribution--------------------------------
parser.add_argument('--tdgia', type=bool, default=False, help="boolean stating whether to use TDGIA's injected data")
parser.add_argument('--save_params', type=bool, default=False, help="boolean stating whether to save the parameters of the model")
parser.add_argument('--save', type=str, default="default", help="save path")
parser.add_argument('--model', type=str, default="gcn_nifa", help="model name")
#-----------------------------------------------------------------------------------

args = parser.parse_args()
print(args)


# -----------------------------------our_contribution--------------------------------------
import os

if args.save=="default":
    args.save='../TDGIA/pretrained_GCNs/' + args.model + '_' + args.dataset

if not os.path.exists(args.save):
    os.mkdir(args.save)

# -----------------------------------main------------------------------------------ 

device = torch.device("cuda", args.device)

if args.before:
    B_ACC = {model:[] for model in args.models}
    B_SP = {model:[] for model in args.models}
    B_EO = {model:[] for model in args.models}
A_ACC = {model:[] for model in args.models}
A_SP = {model:[] for model in args.models}
A_EO = {model:[] for model in args.models}


seed = 42

for i in range(args.n_times):
# -----------------------------------our_contribution--------------------------------------
    # Added a seed for reproducibility:
    set_random_seeds(seed + i)
#------------------------------------------------------------------------------------------

    g, index_split, _, _ = load_data(args.dataset, args)
    g = g.to(device)
    in_dim = g.ndata['feature'].shape[1]
    hid_dim = args.hid_dim
    out_dim = max(g.ndata['label']).item() + 1
    label = g.ndata['label']

    if args.before:
        for model in args.models:
            victim_model = VictimModel(in_dim, hid_dim, out_dim, device, name=model)
            victim_model.optimize(g, index_split, args.epochs, args.lr, args.patience)
            acc, sp, eo = victim_model.eval(g, index_split, args.sensitive_attr_mode)
            B_ACC[model].append(acc)
            B_SP[model].append(sp)
            B_EO[model].append(eo)
# -----------------------------------our_contribution--------------------------------------
            if args.save_params:
                torch.save(victim_model.state_dict(), args.save + "/" + str(i))
#------------------------------------------------------------------------------------------
    
    start_time = time.time()
    attacker = Attacker(g, in_dim, hid_dim, out_dim, device, args)
    g_attack, uncertainty = attacker.attack(g, index_split)  # uncertainty shape: [n_nodes]
    end_time = time.time()
    print(">> Finish attack, cost {:.4f}s.".format(end_time-start_time))
    # dgl.save_graph(g_attack, index_split)
    # import pdb; pdb.set_trace()

    dgl.save_graphs(f"../data/{args.dataset}_poisoned.bin", [g])

# -----------------------------------our_contribution--------------------------------------
    if args.tdgia:

        # Load the injected data from TDGIA
        adj, features, labels, sensitive = load_bin(args.dataset)
        inc_adj, inc_feat = load_injected_data(args.dataset)
        adj, features = combine_features(adj, features, inc_adj, inc_feat)

        # Create the graph with the injected data
        g_attack = to_dgl_graph(adj, features)

        # Add the labels and sensitive attributes to the graph
        g_attack.ndata['label'] = torch.full((g_attack.num_nodes(),), -1)
        g_attack.ndata['label'][:labels.shape[0]] = torch.tensor(labels)
        g_attack.ndata['sensitive'] = torch.full((g_attack.num_nodes(),), -1)
        g_attack.ndata['sensitive'][:labels.shape[0]] = torch.tensor(sensitive)

        # overwrite g_attack with the injected data
        g_attack = g_attack.to(device)
#------------------------------------------------------------------------------------

    for model in args.models:
        victim_model = VictimModel(in_dim, hid_dim, out_dim, device, name=model)
        victim_model.re_optimize(g_attack, uncertainty, index_split, args.epochs, args.lr, args.patience, args.defense)
        acc, sp, eo = victim_model.eval(g_attack, index_split, args.sensitive_attr_mode)
        A_ACC[model].append(acc)
        A_SP[model].append(sp)
        A_EO[model].append(eo)

print("================Finished================")
str_format = lambda x, y: "{:.2f}±{:.2f}".format(np.mean(x[y])*100, np.std(x[y])*100)
for model in args.models:
    print("\033[95m{}\033[0m".format(model))
    if args.before:
        print(">> before acc:{}".format(str_format(B_ACC, model)))
        print(">> before sp:{}".format(str_format(B_SP, model)))
        print(">> before eo:{}".format(str_format(B_EO, model)))
    print(">> after acc:{}".format(str_format(A_ACC, model)))
    print(">> after sp:{}".format(str_format(A_SP, model)))
    print(">> after eo:{}".format(str_format(A_EO, model)))

