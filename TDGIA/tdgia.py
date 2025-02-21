from gcn import *
from models_gcn import *
from extra_utils import *

import pickle as pkl
import numpy as np
import argparse
import os
from torch.autograd import Variable
import dgl
import torch
from scipy.sparse import csr_matrix, vstack, hstack
import random

set_random_seeds()

parser = argparse.ArgumentParser(description="Run SNE.")
parser.add_argument('--dataset',default='pokec_z')
parser.add_argument('--epochs',type=int,default=2001)
parser.add_argument('--models',nargs='+',default=['gcn_nifa'])
parser.add_argument('--modelsextra',nargs='+',default=[])
parser.add_argument('--modelseval',nargs='+',default=['gcn_lm','graphsage_norm','sgcn','rgcn','tagcn','appnp','gin'])
#extra models are only used for label approximation.
parser.add_argument('--gpu', type=int, default='0')
parser.add_argument('--strategy',default='gia')
parser.add_argument('--test_rate',type=int,default=0)
parser.add_argument('--test',type=int,default=50000)
parser.add_argument('--lr',type=float,default=1)
parser.add_argument('--step',type=float,default=0.2)
parser.add_argument('--weight1',type=float,default=0.9)
parser.add_argument('--weight2',type=float,default=0.1)
parser.add_argument('--add_rate',type=float,default=1)
parser.add_argument('--scaling',type=float,default=1)
parser.add_argument('--opt',default="clip")
parser.add_argument('--add_num',type=float,default=500) 
parser.add_argument('--max_connections',type=int,default=100)
parser.add_argument('--connections_self',type=int,default=0)

parser.add_argument('--apply_norm',type=int,default=1)
parser.add_argument('--load',default="default")
parser.add_argument('--sur',default="default")
# also evaluate on surrogate models
parser.add_argument('--save',default="default")
args=parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# -----------------------------------our_contribution---------------------------------------------------------------------------------

if args.dataset in ["pokec_z", "pokec_n", "dblp"]:
    g, adj, features, labels, train_index, val_index, test_index, trainlabels, vallabels, testlabels = load_bins(args.dataset, args.gpu)
    device = torch.device("cuda", 0)

#-------------------------------------------------------------------------------------------------------------------------------------

if args.strategy=="gia":
    opt="sin"

#force it to use "clip"
if args.opt=="fclip":
    opt="clip"
args.opt=opt
    
print("node num:",len(features))
print("edge num:",adj.sum())
print("val num:",len(val_index))
print("test num:",len(test_index))
print("feats num:",len(features[0]))
print("feat range:",features.min(),features.max())

testindex=test_index
total=len(features)
num_classes=np.max(testlabels)+1
num_features=features.shape[1]

if args.test_rate>0:
    args.test=int(args.test_rate*len(features)/100)
    
add=int(args.add_rate*0.01*total)
if args.add_num>0:
    add=args.add_num

def generateaddon(applabels, culabels, adj, origin, cur, testindex, addmax=500, num_classes=18, connect=65, sconnect=20, strategy='random'):
    weight1 = args.weight1
    weight2 = args.weight2
    newedgesx = []
    newedgesy = []
    newdata = []
    thisadd = 0
    num_test = len(testindex)

    addscore = np.zeros(num_test)
    deg = np.array(adj.sum(axis=0))[0] + 1.0
    normadj = GCNadj(adj)
    normdeg = np.array(normadj.sum(axis=0))[0]
    print(culabels[-1])

    for i in range(len(testindex)):
        it = testindex[i]
        label = applabels[it]
        score = culabels[it][label] + 2
        addscore1 = score / deg[it]
        addscore2 = score / np.sqrt(deg[it])
        sc = weight1 * addscore1 + weight2 * addscore2 / np.sqrt(connect + sconnect)
        addscore[i] = sc

    sortedrank = addscore.argsort()
    sortedrank = sortedrank[-addmax * connect:]

    labelgroup = np.zeros(num_classes)
    labelil = [[] for _ in range(num_classes)]
    random.shuffle(sortedrank)
    for i in sortedrank:
        label = applabels[testindex[i]]
        labelgroup[label] += 1
        labelil[label].append(i)

    pos = np.zeros(num_classes)
    print(labelgroup)

    for i in range(addmax):
        for j in range(connect):
            smallest = 1
            smallid = 0
            for k in range(num_classes):
                if len(labelil[k]) > 0:
                    if (pos[k] / len(labelil[k])) < smallest:
                        smallest = pos[k] / len(labelil[k])
                        smallid = k

            if len(labelil[smallid]) == int(pos[smallid]):
                continue
            tu = labelil[smallid][int(pos[smallid])]
            pos[smallid] += 1
            x = cur + i
            y = testindex[tu]
            newedgesx.extend([x, y])
            newedgesy.extend([y, x])
            newdata.extend([1, 1])

    islinked = np.zeros((addmax, addmax))
    for i in range(addmax):
        rndtimes = 100
        while np.sum(islinked[i]) < sconnect and rndtimes > 0:
            x = i + cur
            rndtimes = 100
            yy = random.randint(0, addmax - 1)
            while np.sum(islinked[yy]) >= sconnect or yy == i or islinked[i][yy] == 1 and rndtimes > 0:
                yy = random.randint(0, addmax - 1)
                rndtimes -= 1
            if rndtimes > 0:
                y = cur + yy
                islinked[i][yy] = 1
                islinked[yy][i] = 1
                newedgesx.extend([x, y])
                newedgesy.extend([y, x])
                newdata.extend([1, 1])

    thisadd = addmax
    print(thisadd, len(newedgesx))
    add1 = csr_matrix((thisadd, cur))
    add2 = csr_matrix((cur + thisadd, thisadd))
    adj = vstack([adj, add1])
    adj = hstack([adj, add2])

    adj_coo = adj.tocoo()
    adj_row = np.hstack([adj_coo.row, newedgesx])
    adj_col = np.hstack([adj_coo.col, newedgesy])
    adj_data = np.hstack([adj_coo.data, newdata])

    adj = csr_matrix((adj_data, (adj_row, adj_col)), shape=(adj.shape[0], adj.shape[1]))

    return thisadd, adj
    
def getprocessedadj(adj,modeltype,feature=None):

# -----------------------------------our_contribution--------------------------------------
    if modeltype == "gcn_nifa":
        src, dst = adj.nonzero()
        dgl_adj = dgl.graph((src, dst))
        dgl_adj = dgl.add_self_loop(dgl_adj)

        processed_adj = dgl_adj
#---------------------------------------------------------------------------------------------
    
    return processed_adj
    
def getprocessedfeat(feature,modeltype):
    feat=feature+0.0
    return feat 
    
def getresult(adj,features,model,modeltype):
    processed_adj=getprocessedadj(adj,modeltype,feature=features.data.cpu().numpy())
    features=getprocessedfeat(features,modeltype)

# -----------------------------------our_contribution--------------------------------------
    if modeltype=="gcn_nifa":
        g = processed_adj
        features = getprocessedfeat(features, modeltype)

        model.eval()

        g = g.to(device) 
        features = torch.tensor(features).to(device)

        result = model(g, features)

    return result
#---------------------------------------------------------------------------------------

def checkresult(curlabels,testlabels,origin,testindex):
    evallabels=curlabels[testindex]
    tlabels=torch.LongTensor(testlabels).cuda()
    acc=(evallabels==tlabels)
    acc=acc.sum()/(len(testindex)+0.0)
    acc=acc.item()
    return acc
    
def buildtensor(adj):
    sparserow=torch.LongTensor(adj.row).unsqueeze(1)
    sparsecol=torch.LongTensor(adj.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow,sparsecol),1).cuda()
    sparsedata=torch.FloatTensor(adj.data).cuda()
    adjtensor=torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(adj.shape)).cuda()
    return adjtensor
    
def getmultiresult(adj,features,models,modeltype,origin,testindex):

    iw=0

    with torch.no_grad():
        for i in range(len(models)):
            processed_adj=getprocessedadj(adj,modeltype[i],feature=features.data.cpu().numpy())

            if modeltype[i] == 'gcn_nifa':
                g = processed_adj
                features = getprocessedfeat(features, modeltype[i])

                models[i].eval()

                g = g.to(device) 
                features = torch.tensor(features).to(device)

                iza = models[i](g, features)
                iza=F.softmax(iza,dim=-1)
                iw=iza+iw    
    
    surlabel=iw.argmax(-1).cpu().numpy()
    return surlabel


def trainaddon(adj,thisadd,model,modelrank,testlabels,feature,origin,testindex,num_features=100,strategy='random',reallabels=None,maxlim=1,lbth=4,opt="sin"):

    dim2_adj = dgl.from_scipy(adj)  # or use dgl.graph if you're working with a different type of graph
    dim2_adj = dgl.add_self_loop(dim2_adj)
    dim2_adj = dim2_adj.to('cuda')
    
    feature_origin=feature[:origin]
    # import copy
    feature_added=feature[origin:].cpu().data.numpy()
    #revert it back
    
    if opt=="sin":
        feature_added=feature_added/maxlim
        feature_added=np.arcsin(feature_added)
    feature_added=Variable(torch.FloatTensor(feature_added).cuda(),requires_grad=True)
    add=torch.randn((thisadd,num_features)).cuda()
            
    addontensor=Variable(add)
    addontensor.requires_grad=True
    optimizer=torch.optim.Adam([{'params':[addontensor,feature_added]}], lr=args.lr)
    optimizer.zero_grad()
    best_val=1
    testlabels=torch.LongTensor(testlabels).cuda()
    ep=args.epochs
    if (thisadd+1)*50+1<ep:
        ep=(thisadd+1)*50+1
    if strategy=="degms":
        ep=args.epochs*1.25

    for epoch in range(ep):
      
      i=epoch%(len(model))
      model[i].eval()
      feature_orc=feature_origin+0.0
    
      if modelrank[i] in ['gcn_nifa']:
        adjtensor=dim2_adj

      if opt=="sin":
        feature_add=torch.sin(feature_added)*maxlim
        addontenso=torch.sin(addontensor)*maxlim
      featuretensor=torch.cat((feature_orc,feature_add,addontenso),0)
    
# -----------------------------------our_contribution--------------------------------------
      if modelrank[i] == 'gcn_nifa':
        out1 = model[i](adjtensor, featuretensor, dropout=0)

# -----------------------------------------------------------------------------------------
    
      testout1=out1[testindex]
      loss=nn.CrossEntropyLoss(reduction='none')
      l2=loss(testout1,testlabels)
      if opt=="sin":
        l2=F.relu(-l2+lbth)**2

      l=torch.mean(l2)#+torch.mean(l3)*25
        
      if epoch%75==0:
          
          testoc=testout1.argmax(1)
          acc=(testoc==testlabels)
          acc=acc.sum()/(len(testindex)+0.0)
          acc=acc.item()
          
          print("epoch:",epoch,"loss:",l," acc_tag:",acc)
          
          if reallabels is not None:
            with torch.no_grad():
                tlabels=torch.LongTensor(reallabels).cuda()
                acc=(testoc==tlabels)
                acc=acc.sum()/(len(testindex)+0.0)
                acc=acc.item()
              
                print("epoch:",epoch,"loss:",l," acc_tag:",acc)
                curlabels=getresult(adj,featuretensor,model[i],modelrank[i])
                curlabels=curlabels.argmax(1)
                result=checkresult(curlabels,reallabels,origin,testindex)
                result2=checkresult(curlabels,testoc.data.cpu().numpy(),origin,testindex)
                print(result,result2)
            
          best_addon=featuretensor[origin:].cpu().data
    
      optimizer.zero_grad()
      l.backward()
      
      optimizer.step()
    return best_addon
    
num=0
models=[]
load_str=""
mdsrank=[]
num_models=len(args.models)
args.models.extend(args.modelsextra)
for name in args.models:
    exec("models.append("+name+"("+str(num_features)+','+str(num_classes)+").cuda())")
    dir= "pretrained_GCNs/" + name + "_" + args.dataset+ "/0"

    state_dict = torch.load(dir)

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "")  
        new_state_dict[new_key] = value

    torch.save(new_state_dict, dir)

    models[-1].load_state_dict(torch.load(dir))
    mdsrank.append(name)
same=[]

feature=torch.FloatTensor(features)
featurer=Variable(feature,requires_grad=True).cuda()
mds=[]
for j in range(len(models)):
    mds.append(models[j])
    
print(len(mds))

prlabel=getmultiresult(adj,featurer,mds,mdsrank,total,args.test)
print((prlabel[testindex]==testlabels).sum())

num=0
models=models[:num_models]
mds=mds[:num_models]
feature_origin=feature*1.0
while num<add:
    featurer=Variable(feature,requires_grad=True).cuda()
    # start with 0, to shape it.
    with torch.no_grad():
        curlabels=F.softmax(getresult(adj,featurer,models[0],mdsrank[0]),dim=1)

        for j in range(1,len(models)):
            curlabels+=F.softmax(getresult(adj,featurer,models[j],mdsrank[j]),dim=1)

    curlabels=1/len(models)*curlabels
        
        
    addmax=int(add-num)
    if (addmax>add*args.step):
        addmax=int(add*args.step)
    thisadd,adj_new=generateaddon(prlabel,curlabels,adj,total,total+num,testindex,sconnect=args.connections_self,addmax=addmax,num_classes=num_classes,connect=args.max_connections,strategy=args.strategy)
    if thisadd==0:
        thisadd,adj_new=generateaddon(prlabel,curlabels,adj,total,total+num,testindex,sconnect=args.connections_self,addmax=addmax,num_classes=num_classes,connect=args.max_connections,strategy=args.strategy)
    if num<add:
        num+=thisadd
        adj=adj_new
        print(thisadd,adj.shape)
        best_addon=trainaddon(adj,thisadd,mds,mdsrank,prlabel[testindex],featurer,total,testindex,num_features=num_features,strategy=args.strategy,reallabels=testlabels,maxlim=args.scaling,opt=args.opt)
        feature=torch.cat((feature_origin,best_addon),0)
    same=[]
    for i in range(len(models)):
        featurer=Variable(feature,requires_grad=True).cuda()
        curlabels=getresult(adj,featurer,models[i],mdsrank[i])
        curlabels=curlabels.argmax(1)
        result=checkresult(curlabels,testlabels,total,testindex)
        same.append(result)
        print(i,same)
        print("bb attack average:",np.average(same),"std:",np.std(same))
    
adj=adj.tocsr()
ad=adj[total:]

if args.save=="default":
    args.save= "tdgia_nodes/" + args.dataset+'_'+args.models[0]

if not os.path.exists(args.save):
    os.mkdir(args.save)
    
pkl.dump(ad,open(args.save+"/adj.pkl","wb+"))
mr=feature.cpu()
mr=mr.numpy()
mr=mr[total:]
        
np.save(args.save+"/feature.npy",mr)
