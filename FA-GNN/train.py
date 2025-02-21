import csv
import random
import time
import argparse
import numpy as np
from random import choice
import torch
import torch.nn.functional as F
import torch.optim as optim
import dgl
from attack.attack import attack
from GCN import GCN

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
'''
            Dataset args
'''
parser.add_argument('--dataset', type=str, default='dblp',
                    choices=['pokec_z', 'pokec_n', 'dblp'])
parser.add_argument('--train_percent_atk', type=float, default=0.5,
                    help='Percentage of labeled data available to the attacker.')
parser.add_argument('--train_percent_gnn', type=float, default=0.5,
                    help='Percentage of labeled data as train set.')
parser.add_argument('--val_percent', type=float, default=0.25,
                    help='Percentage of labeled data as validation set.')
parser.add_argument('--sens_number', type=int, default=200,
                    help='Number of sensitive nodes to sample.')
'''
            Model args
'''
parser.add_argument('--model', type=str, default=['gcn'], nargs='+',
                    choices=['gcn', 'gat', 'gsage', 'fairgnn'])
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--attack_type', type=str, default='fair_attack',
                    choices=['none', 'random', 'dice', 'fair_attack'],
                    help='Adversarial attack type.')
parser.add_argument('--sensitive', type=str, default='region',
                    choices=['gender', 'region'],
                    help='Sensitive attribute of Pokec.')
parser.add_argument("--preprocess_pokec", type=bool, default=False,
                    help="Include only completed accounts in Pokec datasets (only valid when dataset==pokec_n/pokec_z])")
parser.add_argument('--ptb_rate', type=float, nargs='+', default=[0.05],
                    help="Attack perturbation rate [0-1]")
parser.add_argument("--num_layers", type=int, default=2,
                    help="number of hidden layers")
# ----args for FairAttack
parser.add_argument('--direction', type=str, default='y1s1',
                    choices=['y1s1', 'y1s0', 'y0s0', 'y0s1'],
                    help='FairAttack direction')
parser.add_argument('--strategy', type=str, default='DD',
                    choices=['DD', 'DE', 'ED', 'EE'],
                    help='FairAttack strategy indicating [D]ifferent/[E]qual label(y)|sens(s)')
parser.add_argument('--deg', type=int, default=0,  # may not finish on small datasets
                    choices=[0, 1, 2, 3],
                    help='Degree parameter, 0 for not considering degree, '
                         'd(high)>deg*d(low).')
parser.add_argument('--deg_direction', type=str, default='null',
                    choices=['hl', 'lh', 'null'],
                    help='Direction of degree difference, '
                         'hl for (subject-influencer)=(high-low), and vice versa,'
                         'null for not considering degree.')

'''
            Optimization args
'''
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument(
    '--acc',
    type=float,
    default=0.2,
    help='the selected FairGNN accuracy on val would be at least this high')
parser.add_argument(
    '--roc',
    type=float,
    default=0.5,
    help='the selected FairGNN ROC score on val would be at least this high')

parser.add_argument('--n_perturbations', type=int, default=42, 
                    help='Number of edge modifications for the attack.')

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed_set = [42, 0, 1, 2, 100]  

g, adj, features, labels, idx_train_atk, idx_train_gnn, idx_val, idx_test, sens, idx_sens_train, dataset, sens_attr, sens_number = load_data_nifa(args)

rate = calc_rate(labels,args)
args.ptb_rate = [rate]


FINAL_RESULT = []
ACC_L = []
SP_L = []
EO_L = []

for ptb_rate in args.ptb_rate:
    N = len(seed_set)
    for repeat in range(N):
        seed = seed_set[repeat]
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)

        g, adj, features, labels, idx_train_atk, idx_train_gnn, idx_val, idx_test, sens, idx_sens_train, dataset, sens_attr, sens_number = load_data_nifa(args)

        if args.attack_type != 'none':
            adj = attack(args, ptb_rate, adj, features, labels, sens, idx_train_atk, idx_val, idx_test,
                            seed, dataset, sens_attr, idx_sens_train)
            check_dataset(dataset, adj, labels, sens, idx_train_gnn, idx_val, idx_test)

        if sens_attr:
            sens[sens > 0] = 1
        idx_train = idx_train_gnn

        g = g.to(device)
        model = GCN(g,
                    features.shape[1],  
                    args.hidden,
                    1,
                    args.num_layers,
                    F.relu,
                    args.dropout
                )
        if args.cuda:
            model.cuda()
            features = features.cuda()
            # adj = adj.cuda()
            sens = sens.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()
            idx_sens_train = idx_sens_train.cuda()

        optimizer = optim.Adam(model.parameters(),
                                lr=args.lr, weight_decay=args.weight_decay)
        loss_fcn = torch.nn.BCEWithLogitsLoss()

        # Train model
        t_total = time.time()
        vali_max = [0, [0, 0, 0, 0, 0, 0], [100, 100, 100], -1]
        loss_all = []
        for epoch in range(args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            output = model(features)
            loss_train = loss_fcn(
                output[idx_train],
                labels[idx_train].unsqueeze(1).float())
            loss_all.append(loss_train.detach().cpu().item())
            acc_train, roc_train, _, _, _, _ = classification_metrics(
                output[idx_train], labels[idx_train])
            # _, _, _, _ = fair_metric(
            #     labels, output, idx_train, sens, 'train')
            loss_train.backward()
            optimizer.step()

            if not args.fastmode:
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                model.eval()
                output = model(features)

            acc_val, roc_val, _, _, _, _ = classification_metrics(
                output[idx_val], labels[idx_val])
            # _,_,_,_ = fair_metric(
            #     labels, output, idx_val, sens, 'val')
            acc_test, roc_test, p, r, maf1_test, mif1_test = classification_metrics(
                output[idx_test], labels[idx_test])
            
            _, _, eq_odds, middle_results = fair_metric(
                labels, output, idx_test, sens, 'test')
        
            output = output.squeeze()
            pred = (output > 0).type_as(labels)

            spi, eo = fair_matrix(pred, labels, sens, idx_test)
            spi = torch.tensor(spi).abs().mean().item()
            eo = torch.tensor(eo).abs().mean().item()

            if acc_val > args.acc:  # and roc_val > args.roc:
                if acc_val > vali_max[0]:
                    vali_max = [
                        acc_val, [
                            acc_test.item(), roc_test, p, r, maf1_test, mif1_test], [
                            spi, eo, eq_odds], epoch + 1, middle_results]

        FINAL_RESULT.append(list(vali_max))
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        ACC_L.append(vali_max[1][0])
        SP_L.append(vali_max[2][0])
        EO_L.append(vali_max[2][1])

        print('============performace on test set=============')
        print("Test:",
                "accuracy: {:.4f}".format(vali_max[1][0]),
                "auc: {:.4f}".format(vali_max[1][1]),
                "sp:{}".format(vali_max[2][0]),
                "eo:{}".format(vali_max[2][1]))
        print("===============================================")
print("Dataset: ", args.dataset)
str_format = lambda x: "{:.2f}±{:.2f}".format(np.mean(x) * 100, np.std(x) * 100)
print(">> acc:{}".format(str_format(ACC_L)))
print(">> sp:{}".format(str_format(SP_L)))
print(">> eo:{}".format(str_format(EO_L)))
print("===============================================")