# Reassessing Fairness: A Reproducibility Study of NIFA’s Impact on GNN Models

This repository presents a reproduction of [Are Your Models Still Fair? Fairness Attacks on
Graph Neural Networks via Node Injections](https://arxiv.org/abs/2406.03052v2). 

## Requirements

Run this command to install the requirements in a conda environment:

```setup
conda env create --name env_nifa --file=env_nifa.yml
```

The FA-GNN code requires a different environement to work which can be installed using this command:
```
conda env create --name env_FAGNN --file=env_FAGNN.yml
```


## Datasets & Processed files

- Due to size limitation, the processed datasets are stored in  [google drive](https://drive.google.com/file/d/1WJYj8K3_H3GmJg-RZeRsJ8Z64gt3qCnq/view?usp=drive_link) as `data.zip`. The datasets include Pokec-z, Pokec-n and DBLP. 

- Download and unzip the `data.zip`, after adding the data the directory tree should be the following:

  ```
    .
    ├── code
    ├── data
    ├── FA-GNN
    ├── FairGNN
    ├── FairSIN
    ├── FairVGNN
    ├── TDGIA
    ├── .gitignore
    ├── analysis.ipynb
    ├── env_FAGNN.yml
    ├── env_nifa.yml
    ├── README.md
    └── results.ipynb
  ```

## Reproducing the experiments

To compute the results you can either run the rusults.ipynb or manually run the commands stated below.

### The following lines reproduce the evaluation of the four classic GNN models (Table 1.)
Note: make sure you are in the code folder

```
python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device 0 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'

python main.py --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 50 --before --device 1 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'

python main.py --dataset dblp --alpha 0.1 --beta 8 --node 32 --edge 24 --epochs 500 --before --device 2 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'
```

### The following lines reproduce the evaluation of FairGNN (Table 1.)
Note: make sure you are in the FairGNN folder

```
python train_fairGNN.py --seed=42 --model=GAT --sens_number=200  --num-hidden=128 --num_layers=2 --dataset=<DATASET> --alpha=4 --beta=0.01 --n_times=5 --dropout=0.5
python train_fairGNN.py --seed=42 --model=GAT --sens_number=200 --num-hidden=128 --num_layers=2 --dataset=pokec_n --alpha=4 --beta=0.01 --n_times=5 --dropout=0.5 --poisoned

python train_fairGNN.py --seed=42 --model=GAT --sens_number=200 --num-hidden=128 --num_layers=2 --dataset=pokec_z --alpha=4 --beta=0.01 --n_times=5 --dropout=0.5
python train_fairGNN.py --seed=42 --model=GAT --sens_number=200 --num-hidden=128 --num_layers=2 --dataset=pokec_z --alpha=4 --beta=0.01 --n_times=5 --dropout=0.5 --poisoned

python train_fairGNN.py --seed=42 --model=GAT --sens_number=200 --num-hidden=128 --num_layers=2 --dataset=dblp --acc=0.93 --alpha=4 --beta=0.01 --n_times=5 --dropout=0.5
python train_fairGNN.py --seed=42 --model=GAT --sens_number=200 --num-hidden=128 --num_layers=2 --dataset=dblp --acc=0.93 --alpha=4 --beta=0.01 --n_times=5 --dropout=0.5 --poisoned

```

### The following lines reproduce the evaluation of FairVGNN (Table 1.)
Note: make sure you are in the FairVGNN folder


```
python fairvgnn.py --dataset='pokec_z' --encoder='GCN' --runs=5 --alpha=0.5 --prop='spmm' --hidden=128
python fairvgnn.py --dataset='pokec_z' --encoder='GCN' --runs=5 --alpha=0.5 --prop='spmm' --hidden=128 --poisoned

python fairvgnn.py --dataset='pokec_n' --encoder='GCN' --runs=5 --alpha=0.5 --prop='spmm' --hidden=128
python fairvgnn.py --dataset='pokec_n' --encoder='GCN' --runs=5 --alpha=0.5 --prop='spmm' --hidden=128 --poisoned

python fairvgnn.py --dataset='dblp' --encoder='GCN' --runs=5 --alpha=0.5 --prop='spmm' --hidden=128
python fairvgnn.py --dataset='dblp' --encoder='GCN' --runs=5 --alpha=0.5 --prop='spmm' --hidden=128 --poisoned
```

### The following lines reproduce the evaluation of FairSIN (Table 1.)
Note: make sure you are in the FairSIN folder

```
python in-train.py --dataset='pokec_n' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=128 --epoch=100 --c_lr=0.001 --e_lr=0.001 --d_lr=0.001 --delta=4 --d='yes'
python in-train.py --dataset='pokec_n' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=128 --epoch=100 --c_lr=0.001 --e_lr=0.001 --d_lr=0.001 --delta=4 --d='yes' --poisoned

python in-train.py --dataset='pokec_z' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=128 --epoch=100 --c_lr=0.001 --e_lr=0.001 --d_lr=0.001 --delta=4 --d='yes'
python in-train.py --dataset='pokec_z' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=128 --epoch=100 --c_lr=0.001 --e_lr=0.001 --d_lr=0.001 --delta=4 --d='yes' --poisoned

python in-train.py --dataset='dblp' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=128 --epoch=100 --c_lr=0.001 --e_lr=0.001 --d_lr=0.001 --delta=4 --d='yes'
python in-train.py --dataset='dblp' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=128 --epoch=100 --c_lr=0.001 --e_lr=0.001 --d_lr=0.001 --delta=4 --d='yes' --poisoned
```

### Use the following cells to pretrain GCN for further use with TDGIA (optional)
Note: pretrained models are already provided

```
python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --n_times 5 --before --device 0 --models 'GCN' --save_params True
python main.py --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 50 --n_times 5 --before --device 0 --models 'GCN' --save_params True
python main.py --dataset dblp --alpha 0.1 --beta 8 --node 32 --edge 24 --n_times 5 --epochs 500 --before --device 0 --models 'GCN' --save_params True
```

### Use the following cells to create new injected nodes and edges using the TDGIA baseline attack
Note: make sure that you are in the TDGIA folder

```
python tdgia.py --dataset pokec_z --add_num 102 --max_connections 50 --models gcn_nifa
python tdgia.py --dataset pokec_n --add_num 87 --max_connections 50 --models gcn_nifa
python tdgia.py --dataset dblp --add_num 32 --max_connections 24 --models gcn_nifa
```

### The following cells reproduce the results from the TDGIA baseline attack (Table 2.)
Note: make sure that you are in the code folder

```
python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --n_times 5 --before --device 0 --models 'GCN' --tdgia True

python main.py --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 50 --n_times 5 --before --device 0 --models 'GCN' --tdgia True

python main.py --dataset dblp --alpha 0.1 --beta 8 --node 32 --edge 24 --n_times 5 --epochs 500 --before --device 0 --models 'GCN' --tdgia True

```

### The following cells reproduce the results from the FA-GNN baseline attack (Table 2.)
Note: make sure that you are in the FA-GNN folder

```
python train.py --dataset pokec_z --model gcn --attack_type fair_attack --direction y1s1 --sensitive region --strategy DD --hidden 128 --sens_number 200

python train.py --dataset pokec_n --model gcn --attack_type fair_attack --direction y1s1 --sensitive region --strategy DD --hidden 128 --sens_number 200

python train.py --dataset dblp --model gcn --attack_type fair_attack --direction y1s1 --sensitive gender --strategy DD --hidden 128 --sens_number 1000
 
```

### The following cells output the results from the Multi-class Sensitive Attribute Dataset (Table 3.)
Note: make sure that you are in the code folder

```
python main.py --pokec_age_bin --sensitive_attr_mode OvA --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --n_times 5 --before --device 0 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'

python main.py --pokec_age_bin --sensitive_attr_mode OvA --dataset pokec_n --alpha 0.01 --beta 4 --node 102 --edge 50 --n_times 5 --before --device 0 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'

python main.py --pokec_age_bin --sensitive_attr_mode OvO --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --n_times 5 --before --device 0 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'

python main.py --pokec_age_bin --sensitive_attr_mode OvO --dataset pokec_n --alpha 0.01 --beta 4 --node 102 --edge 50 --n_times 5 --before --device 0 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'
``` 
