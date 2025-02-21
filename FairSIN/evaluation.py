import torch.nn.functional as F
import torch
from sklearn.metrics import f1_score, roc_auc_score
from utils import fair_metric, fair_matrix

def evaluate(x, classifier, hp, encoder, data, args):
    classifier.eval()
    encoder.eval()

    with torch.no_grad():
        h = encoder(data.x, data.edge_index, data.adj)
        output = classifier(h)

    accs, auc_rocs, F1s, paritys, equalitys = {}, {}, {}, {}, {}

    # pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
    # pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)

    pred_val = output[data.val_mask].argmax(dim=1)  # Shape: [val_size]
    pred_test = output[data.test_mask].argmax(dim=1)  # Shape: [test_size]

    # accs['val'] = pred_val.eq(
    #     data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    # accs['test'] = pred_test.eq(
    #     data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    
    accs['val'] = torch.eq(pred_val,  data.y[data.val_mask]).sum().item() / len(data.val_mask)
    accs['test'] = torch.eq(pred_test,  data.y[data.test_mask]).sum().item() / len(data.test_mask)

    F1s['val'] = f1_score(data.y[data.val_mask].cpu(
    ).numpy(), pred_val.cpu().numpy())

    F1s['test'] = f1_score(data.y[data.test_mask].cpu(
    ).numpy(), pred_test.cpu().numpy())

    probs_val = F.softmax(output[data.val_mask], dim=1)[:, 1].cpu().numpy()
    probs_test = F.softmax(output[data.test_mask], dim=1)[:, 1].cpu().numpy()
    auc_rocs['val'] = roc_auc_score(data.y[data.val_mask].cpu().numpy(), probs_val)
    auc_rocs['test'] = roc_auc_score(data.y[data.test_mask].cpu().numpy(), probs_test)

    # auc_rocs['val'] = roc_auc_score(
    #     data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
    # auc_rocs['test'] = roc_auc_score(
    #     data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())

    # paritys['val'], equalitys['val'] = fair_metric(pred_val.cpu().numpy(), data.y[data.val_mask].cpu(
    # ).numpy(), data.sens[data.val_mask].cpu().numpy())

    # paritys['test'], equalitys['test'] = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu(
    # ).numpy(), data.sens[data.test_mask].cpu().numpy())

    label = data.y
    label = label.long()

    pred = output.argmax(dim=1)

    par, eq = fair_matrix(pred, label, data.sens, data.val_mask)
    paritys['val'] = torch.tensor(par).abs().mean().item()
    equalitys['val'] = torch.tensor(eq).abs().mean().item()

    par, eq = fair_matrix(pred, label, data.sens, data.test_mask)
    paritys['test'] = torch.tensor(par).abs().mean().item()
    equalitys['test'] = torch.tensor(eq).abs().mean().item()


    return accs, auc_rocs, F1s, paritys, equalitys
