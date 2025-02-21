# TDGIA Attack

This repository contains code for running the experiments on the TDGIA baseline attack.

## Prerequisites

Ensure you have the necessary dependencies installed, which were mentioned in the main folder.

## Running TDGIA

The following commands need to be run for creating the new nodes using the TDGIA attack.

```bash
python tdgia.py --dataset pokec_z --add_num 102 --max_connections 50 --models gcn_nifa
python tdgia.py --dataset pokec_n --add_num 87 --max_connections 50 --models gcn_nifa
python tdgia.py --dataset dblp --add_num 32 --max_connections 24 --models gcn_nifa
```

## Running Additional Commands

After completing the steps above, navigate to the code folder and run the next set of commands.

```bash
cd ../code
python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --n_times 5 --before --device 0 --models 'GCN' --tdgia True
python main.py --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 50 --n_times 5 --before --device 0 --models 'GCN' --tdgia True
python main.py --dataset dblp --alpha 0.1 --beta 8 --node 32 --edge 24 --n_times 5 --epochs 500 --before --device 0 --models 'GCN' --tdgia True
```