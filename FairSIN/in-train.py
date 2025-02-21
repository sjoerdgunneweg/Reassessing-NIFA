from dataset import *
from model import *
from utils import *
from evaluation import *
import argparse
from tqdm import tqdm
from torch import tensor
import warnings
warnings.filterwarnings('ignore')
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import time

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
 
def run(data, args):
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()

    acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(
        args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)

    data = data.to(args.device)

    discriminator = MLP_discriminator(args).to(args.device)
    optimizer_d = torch.optim.Adam([
        dict(params=discriminator.lin.parameters(), weight_decay=args.d_wd)], lr=args.d_lr)

    classifier = MLP_classifier(args).to(args.device)
    optimizer_c = torch.optim.Adam([
        dict(params=classifier.lin.parameters(), weight_decay=args.c_wd)], lr=args.c_lr)

    if(args.encoder == 'MLP'):
        encoder = MLP_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.lin.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
    elif(args.encoder == 'GCN'):
        if args.prop == 'scatter':
            encoder = GCN_encoder_scatter(args).to(args.device)
        else:
            encoder = GCN_encoder_spmm(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.lin.parameters(), weight_decay=args.e_wd),
            dict(params=encoder.bias, weight_decay=args.e_wd)], lr=args.e_lr)
    elif(args.encoder == 'GIN'):
        encoder = GIN_encoder(args).to(args.device) 
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.conv.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
    elif(args.encoder == 'SAGE'):
        encoder = SAGE_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.conv1.parameters(), weight_decay=args.e_wd),
            dict(params=encoder.conv2.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)

    
    # examine if the file exists
    if os.path.isfile(args.dataset+'_hadj.pt'):
        print('########## sample already done #############')
        # load data
        new_adj = torch.load(args.dataset+'_hadj.pt')
    else:
        # pretrain neighbor predictor
        data.adj = data.adj - sp.eye(data.adj.shape[0])
        # neighbor select
        print('sample begin.')
        # compute heterogeneous neighbor
        new_adj = torch.zeros((data.adj.shape[0], data.adj.shape[0])).int()
        for i in tqdm(range(data.adj.shape[0])):
            # get all the neighbor nodes' index of node i
            neighbor = torch.tensor(data.adj[i].nonzero()).to(args.device)
            # filter out all the neighbor nodes with different sensitive atrribute from node i 
            mask = (data.sens[neighbor[1]] != data.sens[i])
            h_nei_idx = neighbor[1][mask]
            new_adj[i, h_nei_idx] = 1
        print('select done.')
        # save data
        torch.save(new_adj, args.dataset+'_hadj.pt')
    
    c_X = data.x
    new_adj = new_adj.cpu()
    # compute degree matrix and heterogeneous neighbor's feature 
    deg = np.sum(new_adj.numpy(), axis=1)
    deg = torch.from_numpy(deg).cpu()
    indices = torch.nonzero(new_adj)
    values = new_adj[indices[:, 0], indices[:, 1]]
    mat = torch.sparse_coo_tensor(indices.t(), values, new_adj.shape).float().cpu()
    h_X = torch.spmm(mat,(data.x).cpu()) / deg.unsqueeze(-1) 
    # examine if any row contains NaN 
    mask = torch.any(torch.isnan(h_X), dim=1)
    # delete rows containing NaN 
    h_X = h_X[~mask].to(args.device)
    c_X = c_X[~mask].to(args.device)
    print('node avg degree:',data.edge_index.shape[1]/data.adj.shape[0],' heteroneighbor degree mean:',deg.float().mean(),' node without heteroneghbor:',(deg == 0).sum())
    deg_norm = deg
    deg_norm[deg_norm == 0] = 1
    deg_norm = deg_norm.to(args.device)
    
    model = MLP(len(data.x[0]),args.hidden,len(data.x[0])).to(args.device)
    optimizer = torch.optim.Adam([
            dict(params=model.parameters(), weight_decay=0.001)], lr=args.m_lr)

    from sklearn.model_selection import train_test_split

    indices = np.arange(c_X.shape[0]) # help for check the index after split
    [indices_train, indices_test, y_train, y_test] = train_test_split(indices, indices, test_size=0.1)
    X_train, X_test, y_train, y_test = c_X[indices_train], c_X[indices_test], h_X[indices_train], h_X[indices_test]

    for count in range(args.runs):
        seed_everything(count + args.seed)
        print(f"Running {count}...")
        discriminator.reset_parameters()
        classifier.reset_parameters()
        encoder.reset_parameters()
        model.reset_parameters()

        best_val_tradeoff = 0
        best_val_loss = math.inf
            
        for epoch in range(0, args.epochs):
            # train mlp
            for m_epoch in range(0, args.m_epoch):                
                model.train() # prep model for training
                # train the model #
                optimizer.zero_grad()
                output = model(X_train)
                train_loss = torch.nn.functional.mse_loss(output, y_train)
                # backward pass: compute gradient of the loss with respect to model parameters
                train_loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update running training loss

                model.eval()
                output = model(X_test)
                valid_loss = torch.nn.functional.mse_loss(output, y_test)
 
                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_mlp_state = model.state_dict()

            model.load_state_dict(best_mlp_state)
            model.eval()
            
            # train classifier
            classifier.train()
            encoder.train()
            for epoch_c in range(0, args.c_epochs):
                optimizer_c.zero_grad()
                optimizer_e.zero_grad()
                
                h = encoder(data.x + args.delta * model(data.x), data.edge_index, data.adj)
                output = classifier(h)

                # modified
                loss_c = F.cross_entropy(
                    output[data.train_mask], data.y[data.train_mask].long().to(args.device))

                # loss_c = F.binary_cross_entropy_with_logits(
                #     output[data.train_mask], data.y[data.train_mask].unsqueeze(1).to(args.device))

                loss_c.backward()

                optimizer_e.step()
                optimizer_c.step()

            if args.d == 'yes':
                # train discriminator to recognize the sensitive group
                discriminator.train()
                encoder.train()
                for epoch_d in range(0, args.d_epochs):
                    optimizer_d.zero_grad()
                    optimizer_e.zero_grad()
                    optimizer.zero_grad()

                    h = encoder(data.x + args.delta * model(data.x), data.edge_index, data.adj)
                    output = discriminator(h)

                    # modified
                    sens = data.x[:, args.sens_idx].float()
                    sens = (sens - sens.min()) / (sens.max() - sens.min())

                    loss_d = criterion(output.view(-1),
                                        sens)

                    loss_d.backward()
                    optimizer_d.step()
                    optimizer_e.step()
                    optimizer.step()

            # evaluate classifier
            accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate(
                data.x, classifier, discriminator, encoder, data, args)

            if  auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
                test_acc = accs['test']
                test_auc_roc = auc_rocs['test']
                test_f1 = F1s['test']
                test_parity, test_equality = tmp_parity['test'], tmp_equality['test']
                # print('best_val_tradeoff',epoch)
                best_val_tradeoff = auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val'])
        
        acc[count] = test_acc
        f1[count] = test_f1
        auc_roc[count] = test_auc_roc
        parity[count] = test_parity
        equality[count] = test_equality

    return acc, f1, auc_roc, parity, equality

if __name__ == '__main__':
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='german')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--d_epochs', type=int, default=5)
    parser.add_argument('--c_epochs', type=int, default=10)
    parser.add_argument('--d_lr', type=float, default=0.01)
    parser.add_argument('--d_wd', type=float, default=0)
    parser.add_argument('--c_lr', type=float, default=0.01)
    parser.add_argument('--c_wd', type=float, default=0)
    parser.add_argument('--e_lr', type=float, default=0.01)
    parser.add_argument('--e_wd', type=float, default=0)
    parser.add_argument('--prop', type=str, default='scatter')       
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--encoder', type=str, default='GIN')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--delta', type=float, default=5)
    parser.add_argument('--m_epoch', type=int, default=20)
    parser.add_argument('--d', type=str, default='no')
    parser.add_argument('--m_lr', type=float, default=0.1)
    parser.add_argument('--poisoned', action='store_true', help="Enable poisoning (default: False)")

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    import dgl
    import scipy.sparse as sp

# -------------------------------------------------------------------------------------------------------

    def get_dataset(dataname, poisoned=False):
        if poisoned:
            glist, _ = dgl.load_graphs(f"../data/{dataname}_poisoned.bin")
        else:
            glist, _ = dgl.load_graphs(f"../data/{dataname}.bin")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        g = glist[0].to(device)

        src, dst = g.edges()
        num_nodes = g.num_nodes()
        adj_norm_sp = sp.coo_matrix(
            (np.ones_like(src.cpu().numpy()), (src.cpu().numpy(), dst.cpu().numpy())),
            shape=(num_nodes, num_nodes)
        ).tocsr()

        features = g.ndata['feature'].to(device)
        labels = g.ndata['label'].to(device)
        sens = g.ndata['sensitive'].to(device)

        train_mask = torch.where(g.ndata['train_index'])[0].to(device)
        val_mask = torch.where(g.ndata['val_index'])[0].to(device)
        test_mask = torch.where(g.ndata['test_index'])[0].to(device)

        if dataname == 'pokec_z':
            sens_idx = 4 #region
        elif dataname == 'pokec_n':
            sens_idx = 4 #region
        elif dataname == 'dblp':
            sens_idx = 2 #gender

        x_max, x_min = torch.max(features, dim=0)[0], torch.min(features, dim=0)[0]
        if dataname != 'dblp': # normalization may cause problem for dblp: model not converge
            norm_features = feature_norm(features)
            norm_features[:, sens_idx] = features[:, sens_idx]
            features = norm_features

        data = Data(
            x=features, 
            edge_index=torch.stack([src, dst]), 
            adj=adj_norm_sp, 
            y=labels.float(), 
            train_mask=train_mask, 
            val_mask=val_mask, 
            test_mask=test_mask, 
            sens=sens
        )
        
        return data, sens_idx, x_min, x_max

# -------------------------------------------------------------------------------------------------------

    if args.poisoned:
        print("========Poisoned========")

    data, args.sens_idx, args.x_min, args.x_max = get_dataset(args.dataset, args.poisoned)
    args.num_features, args.num_classes = data.x.shape[1], 2

    acc, f1, auc_roc, parity, equality = run(data, args)

    print('======' + args.dataset + args.encoder + '======')
    print('Acc:', round(np.mean(acc) * 100,2), '±' ,round(np.std(acc) * 100,2), sep='')
    print('Auc roc:', round(np.mean(auc_roc) * 100,2), '±' ,round(np.std(auc_roc) * 100,2), sep='')
    print('f1:', round(np.mean(f1) * 100,2), '±' ,round(np.std(f1) * 100,2), sep='')
    print('parity:', round(np.mean(parity) * 100,2), '±', round(np.std(parity) * 100,2), sep='')
    print('equality:', round(np.mean(equality) * 100,2), '±', round(np.std(equality) * 100,2), sep='')