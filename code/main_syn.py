import torch
import numpy as np
import argparse
import tqdm
import time

import csv

from org.gesis.model.DPAH import DPAH

from model import VictimModel
from attack import *
from utils import load_data
from dataset import get_graph

parser = argparse.ArgumentParser(description="Fairness Attack Source code")

parser.add_argument('--dataset', default='pokec_z', choices=['pokec_z','pokec_n','dblp', 'synthetic'])

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

parser.add_argument('--device', type=int, default=0, help='device ID for GPU')

parser.add_argument('--explore', action='store_true', help='Enable exploration of settings')
parser.add_argument('--explore_h_mm', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
                    help='List of h_mm values to explore')
parser.add_argument('--explore_h_MM', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
                    help='List of h_MM values to explore')
parser.add_argument('--explore_fm', type=float, nargs='+', default=[0.5], 
                    help='List of fm values to explore')

args = parser.parse_args()
print(args)

# -----------------------------------main------------------------------------------ 

device = torch.device("cuda", args.device)

if args.before:
    B_ACC = {model:[] for model in args.models}
    B_SP = {model:[] for model in args.models}
    B_EO = {model:[] for model in args.models}
A_ACC = {model:[] for model in args.models}
A_SP = {model:[] for model in args.models}
A_EO = {model:[] for model in args.models}

results_file = './exploration_results.csv'

if args.explore:
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['h_mm', 'h_MM', 'fm', 'model', 'accuracy', 'sp', 'eo', 'before'])

        for h_mm in args.explore_h_mm:
            for h_MM in args.explore_h_MM:
                for fm in args.explore_fm:
                    for i in range(args.n_times):
                        print(f"Exploring settings: h_mm={h_mm}, h_MM={h_MM}, fm={fm}")
                        g = DPAH(N=2000, fm=fm, d=0.001, plo_M=2.5, plo_m=2.5, h_MM=h_MM, h_mm=h_mm, verbose=False, seed=1)
                        g, index_split, avg_degree, injected_nodes = get_graph(g, feature_dim=100, proxy_correlation=0.8, args=args)

                        args.node = injected_nodes
                        args.edge = avg_degree

                        # g = DPAH(N=5000, fm=0.5, d=0.001, plo_M=2.5, plo_m=2.5, h_MM=0.5, h_mm=0.5, verbose=False, seed=1)
                        # g, index_split = convert_to_dgl(g, feature_dim=100, proxy_correlation=0.8, args=args)

                        g = g.to(device)
                        in_dim = g.ndata['feature'].shape[1]
                        hid_dim = args.hid_dim
                        out_dim = max(g.ndata['label']).item() + 1
                        label = g.ndata['label']

                        if args.before:
                            for model in args.models:
                                victim_model = VictimModel(in_dim, hid_dim, out_dim, device, name=model)
                                victim_model.optimize(g, index_split, args.epochs, args.lr, args.patience)
                                acc, sp, eo = victim_model.eval(g, index_split)
                                B_ACC[model].append(acc)
                                B_SP[model].append(sp)
                                B_EO[model].append(eo)
                                writer.writerow([h_mm, h_MM, fm, model, acc, sp, eo, True])
                        
                        start_time = time.time()
                        attacker = Attacker(g, in_dim, hid_dim, out_dim, device, args)
                        g_attack, uncertainty = attacker.attack(g, index_split)  # uncertainty shape: [n_nodes]
                        end_time = time.time()
                        print(">> Finish attack, cost {:.4f}s.".format(end_time-start_time))
                        # save_graph(g_attack, index_split)
                        # import pdb; pdb.set_trace()

                        dgl.save_graphs(f"./output/{args.dataset}_poisoned.bin", [g])

                        for model in args.models:
                            victim_model = VictimModel(in_dim, hid_dim, out_dim, device, name=model)
                            victim_model.re_optimize(g_attack, uncertainty, index_split, args.epochs, args.lr, args.patience, args.defense)
                            acc, sp, eo = victim_model.eval(g_attack, index_split)
                            A_ACC[model].append(acc)
                            A_SP[model].append(sp)
                            A_EO[model].append(eo)
                            writer.writerow([h_mm, h_MM, fm, model, acc, sp, eo, False])
                            print(f"After Attack - Model: {model} | Acc: {acc:.4f}, SP: {sp:.4f}, EO: {eo:.4f}")

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

else:
    g = DPAH(N=5000, fm=0.5, d=0.001, plo_M=2.5, plo_m=2.5, h_MM=0.5, h_mm=0.5, verbose=False, seed=1)
    g, index_split, avg_degree, injected_nodes = get_graph(g, feature_dim=100, proxy_correlation=0.8, args=args)

    args.node = injected_nodes
    args.edge = avg_degree

    # g = DPAH(N=5000, fm=0.5, d=0.001, plo_M=2.5, plo_m=2.5, h_MM=0.5, h_mm=0.5, verbose=False, seed=1)
    # g, index_split = convert_to_dgl(g, feature_dim=100, proxy_correlation=0.8, args=args)

    g = g.to(device)
    in_dim = g.ndata['feature'].shape[1]
    hid_dim = args.hid_dim
    out_dim = max(g.ndata['label']).item() + 1
    label = g.ndata['label']

    if args.before:
        for model in args.models:
            victim_model = VictimModel(in_dim, hid_dim, out_dim, device, name=model)
            victim_model.optimize(g, index_split, args.epochs, args.lr, args.patience)
            acc, sp, eo = victim_model.eval(g, index_split)
            B_ACC[model].append(acc)
            B_SP[model].append(sp)
            B_EO[model].append(eo)
    
    start_time = time.time()
    attacker = Attacker(g, in_dim, hid_dim, out_dim, device, args)
    g_attack, uncertainty = attacker.attack(g, index_split)  # uncertainty shape: [n_nodes]
    end_time = time.time()
    print(">> Finish attack, cost {:.4f}s.".format(end_time-start_time))
    # save_graph(g_attack, index_split)
    # import pdb; pdb.set_trace()

    dgl.save_graphs(f"./output/{args.dataset}_poisoned.bin", [g])

    for model in args.models:
        victim_model = VictimModel(in_dim, hid_dim, out_dim, device, name=model)
        victim_model.re_optimize(g_attack, uncertainty, index_split, args.epochs, args.lr, args.patience, args.defense)
        acc, sp, eo = victim_model.eval(g_attack, index_split)
        A_ACC[model].append(acc)
        A_SP[model].append(sp)
        A_EO[model].append(eo)
        print(f"After Attack - Model: {model} | Acc: {acc:.4f}, SP: {sp:.4f}, EO: {eo:.4f}")

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

