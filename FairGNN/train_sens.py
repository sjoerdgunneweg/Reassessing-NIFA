#%%
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy,load_pokec,load_pokec_emb
from models.GCN import GCN
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=41, help='Random seed.')
parser.add_argument('--epochs', type=int, default=800,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--sens_number', type=int, default=200,
                    help="the number of sensitive attributes")
parser.add_argument('--dataset', type=str, default='pokec_z',
                    choices=['pokec_z','pokec_n','nba', 'dblp'])
parser.add_argument('--poisoned', action='store_true', 
                    help="Enable poisoning (default: False)")

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()

#%%
#%%
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
print(args.dataset)

if args.dataset != 'dblp':
    if args.dataset == 'pokec_z':
        dataset = 'region_job'
    else:
        dataset = 'region_job_2'
    sens_attr = "region"
    predict_attr = "I_am_working_in_field"
    # label_number = 500
    sens_number = args.sens_number
    seed = 20
    path="../dataset/pokec/"
    test_idx=False
else:
    dataset = 'dblp'
    sens_attr = "gender"
    predict_attr = "label"
    # label_number = 100
    sens_number = args.sens_number
    seed = 20
    test_idx=False
print(dataset)

import dgl
import scipy.sparse as sp

if args.poisoned:
    glist, _ = dgl.load_graphs(f'../../data/{args.dataset}_poisoned.bin')
else:
    glist, _ = dgl.load_graphs(f"../../data/{args.dataset}.bin")

device = torch.device("cuda", 0)

g = glist[0].to(device)

src, dst = g.edges()
num_nodes = g.num_nodes()
adj = sp.coo_matrix((np.ones_like(src.cpu().numpy()), (src.cpu().numpy(), dst.cpu().numpy())),
                    shape=(num_nodes, num_nodes)).tocsr()

# Convert graph attributes for FairGNN
features = g.ndata['feature'].to(device)
labels = g.ndata['label'].to(device)
sens = g.ndata['sensitive'].to(device)

idx_train = torch.where(g.ndata['train_index'])[0].to(device)
idx_val = torch.where(g.ndata['val_index'])[0].to(device)
idx_test = torch.where(g.ndata['test_index'])[0].to(device)

idx_sens_train = idx_train[sens[idx_train] == 1].to(device)

sens[sens>0]=1
if sens_attr:
    sens[sens>0]=1
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    # adj = adj.cuda()
    sens = sens.cuda()
    # idx_sens_train = idx_sens_train.cuda()
    # idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    sens = sens.cuda()
    idx_sens_train = idx_sens_train.cuda()

from sklearn.metrics import accuracy_score,roc_auc_score,f1_score

# Train model
t_total = time.time()
best_acc = 0.0
best_test = 0.0
for epoch in range(args.epochs+1):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(g, features)
    loss_train = F.binary_cross_entropy_with_logits(output[idx_sens_train], sens[idx_sens_train].unsqueeze(1).float())
    acc_train = accuracy(output[idx_sens_train], sens[idx_sens_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(g, features)
    if epoch%10==0:
        acc_val = accuracy(output[idx_val], sens[idx_val])
        acc_test = accuracy(output[idx_test], sens[idx_test])
        print("Epoch [{}] Test set results:".format(epoch),
            "acc_test= {:.4f}".format(acc_test.item()),
            "acc_val: {:.4f}".format(acc_val.item()))
        if acc_val > best_acc:
            best_acc = acc_val
            best_test = acc_test
            torch.save(model.state_dict(),"./checkpoint/GCN_sens_{}_ns_{}".format(dataset,sens_number))

    checkpoint = torch.load("./checkpoint/GCN_sens_{}_ns_{}".format(dataset, sens_number))
    print("GCN layer 1 weight shape:", checkpoint["body.gc1.weight"].shape)
print("The accuracy of estimator: {:.4f}".format(best_test))

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


# %%
