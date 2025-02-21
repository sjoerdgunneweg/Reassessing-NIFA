#%%
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy,load_pokec, fair_matrix
from models.FairGNN import FairGNN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units of the sensitive attribute estimator')
parser.add_argument('--dropout', type=float, default=.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=4,
                    help='The hyperparameter of alpha')
parser.add_argument('--beta', type=float, default=0.01,
                    help='The hyperparameter of beta')
parser.add_argument('--model', type=str, default="GAT",
                    help='the type of model GCN/GAT')
parser.add_argument('--dataset', type=str, default='pokec_n',
                    choices=['pokec_z','pokec_n','nba', 'dblp'])
parser.add_argument('--num-hidden', type=int, default=64,
                    help='Number of hidden units of classifier.')
parser.add_argument("--num-heads", type=int, default=1,
                        help="number of hidden attention heads")
parser.add_argument("--num-out-heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num-layers", type=int, default=1,
                    help="number of hidden layers")
parser.add_argument("--residual", action="store_true", default=False,
                    help="use residual connection")
parser.add_argument("--attn-drop", type=float, default=.0,
                    help="attention dropout")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")
parser.add_argument('--acc', type=float, default=0.680,
                    help='the selected FairGNN accuracy on val would be at least this high')
parser.add_argument('--roc', type=float, default=0.745,
                    help='the selected FairGNN ROC score on val would be at least this high')
parser.add_argument('--sens_number', type=int, default=200,
                    help="the number of sensitive attributes")
parser.add_argument('--label_number', type=int, default=500,
                    help="the number of labels")
parser.add_argument('--poisoned', action='store_true', 
                    help="Enable poisoning (default: False)")
parser.add_argument('--n_times', type=int, default=5, 
                    help="Number of times to run")


args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

print(args.dataset)
if args.poisoned:
    print("POISONED")

if args.dataset != 'dblp':
    if args.dataset == 'pokec_z':
        dataname = 'region_job'
    else:
        dataname = 'region_job_2'
    sens_attr = "region"
    predict_attr = "I_am_working_in_field"
    label_number = args.label_number
    sens_number = args.sens_number
    seed = 20
    path="../dataset/pokec/"
    test_idx=False

elif args.dataset == 'dblp':
    dataname = 'dblp'
    sens_attr = "gender"
    predict_attr = "label"
    path = "../dataset/dblp/"
    label_number = 1000
    sens_number = args.sens_number
else:
    print("Dataset unknown")

import dgl
import scipy.sparse as sp

# -------------------------------------------------------------------------------------------------------
# Load NIFA datasets
if args.poisoned:
    glist, _ = dgl.load_graphs(f'../data/{args.dataset}_poisoned.bin')
else:
    glist, _ = dgl.load_graphs(f"../data/{args.dataset}.bin")

device = torch.device("cuda", 0)

g = glist[0].to(device)

src, dst = g.edges()
num_nodes = g.num_nodes()
adj = sp.coo_matrix((np.ones_like(src.cpu().numpy()), (src.cpu().numpy(), dst.cpu().numpy())),
                    shape=(num_nodes, num_nodes)).tocsr()
features = g.ndata['feature'].to(device)
labels = g.ndata['label'].to(device)
sens = g.ndata['sensitive'].to(device)

idx_train = torch.where(g.ndata['train_index'])[0].to(device)
idx_val = torch.where(g.ndata['val_index'])[0].to(device)
idx_test = torch.where(g.ndata['test_index'])[0].to(device)

idx_sens_train = idx_train[sens[idx_train] == 1].to(device)

ACC = []
SP = []
EO = []
# -------------------------------------------------------------------------------------------------------

model = FairGNN(nfeat = features.shape[1], args = args)

if args.cuda:
    model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    sens = sens.cuda()
    idx_sens_train = idx_sens_train.cuda()

from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score

for n in range(args.n_times):
    seed = n + args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(seed)

    # Train model
    t_total = time.time()
    best_result = {}
    best_fair = 100

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        model.optimize(g,features,labels,idx_train,sens,idx_sens_train)
        cov = model.cov
        cls_loss = model.cls_loss
        adv_loss = model.adv_loss
        model.eval()
        output,s = model(g, features)
        acc_val = accuracy(output[idx_val], labels[idx_val])
        roc_val = roc_auc_score(labels[idx_val].cpu().numpy(),output[idx_val].detach().cpu().numpy())

        acc_sens = accuracy(s[idx_test], sens[idx_test])

        acc_test = accuracy(output[idx_test], labels[idx_test])
        roc_test = roc_auc_score(labels[idx_test].cpu().numpy(),output[idx_test].detach().cpu().numpy())
        
        pred = (output > 0).type_as(labels)

        parity_val, equality_val = fair_matrix(pred, labels, sens, idx_val)
        parity_val = torch.tensor(parity_val).abs().mean().item()
        equality_val = torch.tensor(equality_val).abs().mean().item()

        parity, equality = fair_matrix(pred, labels, sens, idx_test)
        parity = torch.tensor(parity).abs().mean().item()
        equality = torch.tensor(equality).abs().mean().item()

        if acc_val > args.acc and roc_val > args.roc:
        
            if best_fair > parity_val + equality_val :
                best_fair = parity_val + equality_val

                best_result['acc'] = acc_test.item()
                best_result['roc'] = roc_test
                best_result['parity'] = parity
                best_result['equality'] = equality


    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    ACC.append(acc_test.item())
    SP.append(parity)
    EO.append(equality)

print('============performace on test set=============')
print("Dataset: ", args.dataset)
str_format = lambda x: "{:.2f}±{:.2f}".format(np.mean(x) * 100, np.std(x) * 100)
print(">> acc:{}".format(str_format(ACC)))
print(">> sp:{}".format(str_format(SP)))
print(">> eo:{}".format(str_format(EO)))
print('===============================================')