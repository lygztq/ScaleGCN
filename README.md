# Code for Scale Graph Convolutional Networks (ScaleGCN)

## Dependency

To run code in this repo, please install packages listed below.

```
numpy==1.18.5
scipy==1.5.4
pandas==1.1.4
networkx==2.5
matplotlib==3.3.3
scikit-learn==0.23.2
torch==1.6.0
torchvision==0.7.0
torch-cluster==1.5.8
torch-geometric==1.6.3
torch-scatter==2.0.5
torch-sparse==0.6.8
torch-spline-conv==1.2.0
tqdm==4.46.0
urllib3==1.25.8
```

## Files

```
.
├── experiments             # directory holding experiment scripts
│   ├── citation            # scripts for experiments on citation networks
│   │   └── scalegcn.py
│   ├── ppi                 # scripts for experiments on PPI datasets
│   │   └── scalegcn.py
│   └── reddit              # scripts for experiments on Reddit datasets
│       └── scalegcn.py
├── layers                  # directory holding graph convolution layer modules
│   ├── __init__.py
│   ├── scale_gconv.py      # ScaleGConv, ScaleStarGConv, BiScaleStarGConv modules
│   └── types.py
├── models                  # directory holding graph convolutional network modules
│   ├── __init__.py
│   ├── mlp.py              # helper class: a multi-layer perceptron
│   └── scale_gcn.py        # ScaleGCN, BiScaleGCN modules
├── README.md
├── requirements.txt        # the requirements file used by pip
└── utils                   # some tools used by this code repo
    ├── adj_norms.py
    ├── data_utils.py
    ├── func_utils.py
    ├── __init__.py
    ├── logger.py
    ├── profiler.py
    ├── record_utils.py
    ├── train_utils.py
    └── visualization.py
```

## How to run this code

**Citation Networks**

For experiments on citation dataset, at root directory, run:
```bash
python experiments/citation/scalegcn.py
```
NOTE: you can change hyper-parameters (`dataset`, `num_layers`, etc.) directly in the `experiments/citation/scalegcn.py` script.

**Reddit**

For experiments on Reddit dataset, at root directory, run:
```bash
python experiments/reddit/scalegcn.py
```
NOTE: you can change hyper-parameters (`dropout`, `num_layers`, etc.) directly in the `experiments/reddit/scalegcn.py` script.

**PPI**

For experiments on PPI dataset, at root directory, run:
```bash
python experiments/ppi/scalegcn.py
```
NOTE: you can change hyper-parameters (`dropout`, `num_layers`, etc.) directly in the `experiments/ppi/scalegcn.py` script.
