import os
import time
import torch
import argparse
import logging
from torch_geometric.datasets import Reddit, PPI
from torch_geometric.data import ClusterData, ClusterLoader, Batch, DataLoader

from .adj_norms import adj_normalize_transform
from .func_utils import func_with_parameters


def load_ppi_data(prefix_path, num_clusters=None, batch_size=None):
    trainset = PPI(prefix_path, split="train")
    valset = PPI(prefix_path, split="val")
    testset = PPI(prefix_path, split="test")

    return {"train": trainset, "val": valset, "test": testset}


def load_reddit_data(prefix_path, diag_lambda=None, num_clusters=None, batch_size=None):
    if diag_lambda is None:
        adj_normalization = None
    elif diag_lambda < 0:
        adj_normalization = func_with_parameters(adj_normalize_transform, norm_method="normal")
    else:
        adj_normalization = func_with_parameters(adj_normalize_transform, norm_method="diag_enhance", diag_lambda=diag_lambda)
    dataset = Reddit(prefix_path, pre_transform=adj_normalization)
    return dataset

def norm_path(path):
    return os.path.expanduser(os.path.normpath(path))

def parse_arg():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("-D", "--dataset", type=str, default="Reddit", choices=["Reddit", "PPI"], help="Dataset used for model training")
    parser.add_argument("--dataset_path", type=str, default="~/data", help="Prefix path where downloaded dataset stroed")
    
    # output dir, log and ckpt etc.
    parser.add_argument("-O", "--output", type=str, default="./output", help="Path to store output files (e.g. checkpoints and log file)")
    parser.add_argument("--name", type=str, default="", help="Name of this experiment")
    
    # training setting
    parser.add_argument("--device", type=int, default=0, help="Device Id, -1 for cpu.")
    parser.add_argument("--default_cuda", action="store_true", help="Use default CUDA device")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs to train")
    parser.add_argument("--weight_decay", type=float, default=0., help="Weight decay rate for L2 loss on embedding matrix")
    parser.add_argument("--early_stopping", type=int, default=1000, help="Tolerance for early stopping (# of epochs)")
    parser.add_argument("--num_clusters", type=int, default=50, help="Number of clusters")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of clusters per batch")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (1 - keep_prob)")
    parser.add_argument("--validation", action="store_true", help="Print validation acc after epoch")
    
    # model setting
    parser.add_argument("--hidden", type=int, default=2048, help="Number of units in hidden layer")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of GCN layers")
    parser.add_argument("--diag_lambda", type=float, default=1., help="A positive number for diagonal enhancement, -1 for normal normalization")
    parser.add_argument("--layernorm", action="store_true", help="Whether to use layer normalization")
    parser.add_argument("--multi_label", action="store_true", help="Multilabel or multiclass")
    
    # preprocess
    args = parser.parse_args()
    args.dataset_path = norm_path(args.dataset_path)
    args.output = norm_path(args.output)
    args.device = "cpu" if args.device == -1 else "cuda:{}".format(args.device)
    if args.default_cuda:
        args.device = "cuda"
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available on this machine, using CPU for training instead...")
        args.device = "cpu"
    
    if args.name == "" or args.name is None:
        args.name = "experiment_{}".format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    args.output = os.path.join(args.output, args.name)
    return args
