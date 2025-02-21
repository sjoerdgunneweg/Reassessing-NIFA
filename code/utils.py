import torch
import dgl
import csv
import itertools
import random
import numpy as np

def load_data(dataset, args):
    dataset = dataset.lower()
    assert dataset in ['pokec_z','pokec_n', 'dblp']
    
    glist, _ = dgl.load_graphs(f'../data/{dataset}.bin')
    g = glist[0]

    idx_train = torch.where(g.ndata['train_index'])[0]
    idx_val = torch.where(g.ndata['val_index'])[0]
    idx_test = torch.where(g.ndata['test_index'])[0]

    sens = g.ndata['sensitive']
    idx_sens_train = idx_train[sens[idx_train] == 1]

    # g.ndata.pop('train_index')
    # g.ndata.pop('val_index')
    # g.ndata.pop('test_index')
    index_split = {'train_index': idx_train,
                    'val_index': idx_val,
                    'test_index': idx_test}
    

    # Changing the sensitive attribute to be the age group for testing the multi-class sensitive attribute mode:
    #-----------------------------------------------------------------------------------------------
    if args.pokec_age_bin and ((dataset == 'pokec_z') | (dataset == 'pokec_n')):
        age = g.ndata['feature'][:,3]
        bins = [-float('inf'), 18, 24, 29, 36, float('inf')]
        classes = torch.tensor([0, 1, 2, 3, 4])

        # Digitize the ages based on bins
        age_classes = torch.bucketize(age, torch.tensor(bins), right=False) - 1

        g.ndata['feature'][:,3] = g.ndata['sensitive']
        g.ndata['sensitive'] = age_classes
    #-----------------------------------------------------------------------------------------------

    return g, index_split, sens, idx_sens_train


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

def fair_matrix_multi_class(pred, label, sens, index, mode):
    """Computes the multi-class fairness metrics for the given prediction, label, sensitive attribute and index.

    Args:
        pred: The hard predictions of the model for each node
        label: The ground truth label for each node
        sens: The sensitive attribute for each node
        index: The index of the nodes to consider
        mode: The mode of the multi-class fairness metrics. It can be either 'OvA' or 'OvO'

    Raises:
        NotImplementedError: When the mode is not 'OvA' or 'OvO'

    Returns:
        SP_multiclass: a list with the individual differences that need to be made absolute and averaged to obtain the SP fairness metric.
        EO_multiclass: a list with the individual differences that need to be made absolute and averaged to obtain the EO fairness metric.
    """    
    # One vs All with original metrics:
    SP_multiclass = []
    EO_multiclass = []

    sens_unique_vals = torch.unique(sens)
    sens_unique_vals = sens_unique_vals[sens_unique_vals != -1] # remove -1 because it is the value for injected nodes.

    if mode == 'OvA':
        # loops over all possible sensitive attributes:
        for sensitive_attr in sens_unique_vals:
            sens_ova_binary = (sens==sensitive_attr)

            sp_binary, eo_binary = fair_matrix(pred, label, sens_ova_binary, index)
            SP_multiclass += sp_binary
            EO_multiclass += eo_binary

    elif mode == 'OvO':
        # loops over all possible pairs of sensitive attributes:
        for sens_a, sens_b in itertools.combinations(sens_unique_vals, 2):
            sens_ovo_binary = -torch.ones_like(sens)
            sens_ovo_binary[sens==sens_a] = 1
            sens_ovo_binary[sens==sens_b] = 0
            
            sp_binary, eo_binary = fair_matrix(pred, label, sens_ovo_binary, index)
            SP_multiclass += sp_binary
            EO_multiclass += eo_binary
    else:
        raise NotImplementedError()
    
    return SP_multiclass, EO_multiclass
        



def set_seed(seed):
    """Sets the seed for reproducibility in numpy, random, torch and dgl.

    Args:
        seed: The seed to set for reproducibility.
    """    
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print("cuda is available")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dgl.seed(seed)
    
