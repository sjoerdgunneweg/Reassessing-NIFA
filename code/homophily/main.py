import torch
import numpy as np
import argparse
import tqdm
import time

import csv

from org.gesis.model.DPAH import DPAH

from model import VictimModel
from attack import *
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
parser.add_argument('--seed', type=int, default=42, help='set seed for reproducibility')

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
parser.add_argument('--explore_h_mm', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
parser.add_argument('--explore_h_MM', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
parser.add_argument('--explore_fm', type=float, nargs='+', default=[0.1])
parser.add_argument('--consistent', action='store_true', help='Enable consistent rho')

args = parser.parse_args()
print(args)

# -----------------------------------main------------------------------------------ 

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(seed)

device = torch.device("cuda", args.device)

if args.before:
    B_ACC = {model:[] for model in args.models}
    B_SP = {model:[] for model in args.models}
    B_EO = {model:[] for model in args.models}
A_ACC = {model:[] for model in args.models}
A_SP = {model:[] for model in args.models}
A_EO = {model:[] for model in args.models}

results_file = f'./exploration_results_{args.seed}.csv'
graph_file = f'./graphtype_results_{args.seed}.csv'

if args.explore:
    set_seed(args.seed)
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['h_mm', 'h_MM', 'fm', 'model', 'acc', 'sp', 'eo', 'phase', 'h'])
        for h_mm in args.explore_h_mm:
            for h_MM in args.explore_h_MM:
                for fm in args.explore_fm:
                    print(f"Exploring settings: h_mm={h_mm}, h_MM={h_MM}, fm={fm}")
                    g = DPAH(N=2000, fm=fm, d=0.0015, plo_M=2.5, plo_m=2.5, h_MM=h_MM, h_mm=h_mm, verbose=False, seed=1)
                    g, index_split, avg_degree, injected_nodes = get_graph(g, feature_dim=100, proxy_correlation=0.5, args=args)
                    sens = g.ndata['sensitive']
                    h = dgl.edge_homophily(g, sens)

                    args.node = injected_nodes
                    args.edge = avg_degree

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
                            writer.writerow([h_mm, h_MM, fm, model, acc, (sp * 100), (eo * 100), 'before', h])
                    
                    attacker = Attacker(g, in_dim, hid_dim, out_dim, device, args)
                    g_attack, uncertainty = attacker.attack(g, index_split)  # uncertainty shape: [n_nodes]

                    # dgl.save_graphs(f"./output/{args.dataset}_poisoned.bin", [g])

                    for model in args.models:
                        victim_model = VictimModel(in_dim, hid_dim, out_dim, device, name=model)
                        victim_model.re_optimize(g_attack, uncertainty, index_split, args.epochs, args.lr, args.patience, args.defense)
                        acc, sp, eo = victim_model.eval(g_attack, index_split)
                        A_ACC[model].append(acc)
                        A_SP[model].append(sp)
                        A_EO[model].append(eo)
                        writer.writerow([h_mm, h_MM, fm, model, acc, (sp * 100), (eo * 100), 'after', h])

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
    graph_settings = {
        'Heterophilic': {'h_mm': 0.2, 'h_MM': 0.2, 'rho': 0.2},
        'Hete/Homo': {'h_mm': 0.8, 'h_MM': 0.2, 'rho': 0.4},
        'Neutral': {'h_mm': 0.5, 'h_MM': 0.5, 'rho': 0.5},
        'Homo/Hete': {'h_mm': 0.2, 'h_MM': 0.8, 'rho': 0.8},
        'Homophilic': {'h_mm': 0.8, 'h_MM': 0.8, 'rho': 1.0}
    }

    results = {
        graph: {
            'before': {'acc': [], 'sp': [], 'eo': []},
            'after': {'acc': [], 'sp': [], 'eo': []}
        } 
        for graph, _ in graph_settings.items()
    }

    for i in range(args.n_times):
        set_seed(args.seed + i)
        for graph, params in graph_settings.items():
            h_mm = params['h_mm']
            h_MM = params['h_MM']
            rho = params['rho']

            if args.consistent:
                rho = 0.5

            fm = 0.3

            g = DPAH(N=2000, fm=fm, d=0.0015, plo_M=2.5, plo_m=2.5, h_MM=h_MM, h_mm=h_mm, verbose=False, seed=1)
            g, index_split, avg_degree, injected_nodes = get_graph(g, feature_dim=100, proxy_correlation=rho, args=args)

            args.node = injected_nodes
            args.edge = avg_degree

            g = g.to(device)
            in_dim = g.ndata['feature'].shape[1]
            hid_dim = args.hid_dim
            out_dim = max(g.ndata['label']).item() + 1

            if args.before:
                for model in args.models:
                    victim_model = VictimModel(in_dim, hid_dim, out_dim, device, name=model)
                    victim_model.optimize(g, index_split, args.epochs, args.lr, args.patience)
                    acc, sp, eo = victim_model.eval(g, index_split)
                    results[graph]['before']['acc'].append(acc)
                    results[graph]['before']['sp'].append(sp)
                    results[graph]['before']['eo'].append(eo)

            attacker = Attacker(g, in_dim, hid_dim, out_dim, device, args)
            g_attack, uncertainty = attacker.attack(g, index_split)

            # dgl.save_graphs(f"./output/{args.dataset}_poisoned.bin", [g])

            for model in args.models:
                victim_model = VictimModel(in_dim, hid_dim, out_dim, device, name=model)
                victim_model.re_optimize(g_attack, uncertainty, index_split, args.epochs, args.lr, args.patience, args.defense)
                acc, sp, eo = victim_model.eval(g_attack, index_split)
                results[graph]['after']['acc'].append(acc)
                results[graph]['after']['sp'].append(sp)
                results[graph]['after']['eo'].append(eo)

    with open(graph_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['graph', 'h_mm', 'h_MM', 'fm', 'model', 'mean_acc', 'std_acc', 'mean_sp', 'std_sp', 'mean_eo', 'std_eo', 'phase', 'rho'])

        for graph, _ in graph_settings.items():
            h_mm = graph_settings[graph]['h_mm']
            h_MM = graph_settings[graph]['h_MM']
            rho = graph_settings[graph]['rho']

            if args.consistent:
                rho = 0.5

            if args.before:
                for model in args.models:
                    writer.writerow([
                        graph, h_mm, h_MM, model, fm,
                        np.mean(results[graph]['before']['acc']) * 100, np.std(results[graph]['before']['acc']) * 100,
                        np.mean(results[graph]['before']['sp']) * 100, np.std(results[graph]['before']['sp']) * 100,
                        np.mean(results[graph]['before']['eo']) * 100, np.std(results[graph]['before']['eo']) * 100,
                        'before',
                        rho
                    ])

            for model in args.models:
                writer.writerow([
                    graph, h_mm, h_MM, model, fm,
                    np.mean(results[graph]['after']['acc']) * 100, np.std(results[graph]['after']['acc']) * 100,
                    np.mean(results[graph]['after']['sp']) * 100, np.std(results[graph]['after']['sp']) * 100,
                    np.mean(results[graph]['after']['eo']) * 100, np.std(results[graph]['after']['eo']) * 100,
                    'after',
                    rho
                ])

    print("================Finished================")
    for graph, _ in graph_settings.items():
        print(f"\033[95m{graph}\033[0m")
        if args.before:
            print(">> before acc: {:.2f}±{:.2f}".format(np.mean(results[graph]['before']['acc']) * 100, np.std(results[graph]['before']['acc']) * 100))
            print(">> before sp: {:.2f}±{:.2f}".format(np.mean(results[graph]['before']['sp']) * 100, np.std(results[graph]['before']['sp']) * 100))
            print(">> before eo: {:.2f}±{:.2f}".format(np.mean(results[graph]['before']['eo']) * 100, np.std(results[graph]['before']['eo']) * 100))
        print(">> after acc: {:.2f}±{:.2f}".format(np.mean(results[graph]['after']['acc']) * 100, np.std(results[graph]['after']['acc']) * 100))
        print(">> after sp: {:.2f}±{:.2f}".format(np.mean(results[graph]['after']['sp']) * 100, np.std(results[graph]['after']['sp']) * 100))
        print(">> after eo: {:.2f}±{:.2f}".format(np.mean(results[graph]['after']['eo']) * 100, np.std(results[graph]['after']['eo']) * 100))