from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from typing import Dict, Optional, List
import numpy as np
from torch.functional import Tensor
from torch_geometric.utils import get_laplacian


def sample_per_class(num_classes:int, num_samples_per_class:int, y:torch.Tensor):
    num_nodes = y.size(0)
    mask = torch.full((num_nodes, ), fill_value=False, device=y.device, dtype=torch.bool)
    for c in range(num_classes):
        idx = (y == c).nonzero().view(-1)
        idx = idx[torch.randperm(idx.size(0))][:num_samples_per_class]
        mask[idx] = True
    return mask


def visualize_feature(x:torch.Tensor, y:torch.Tensor, mask:Optional[torch.Tensor]=None,
                      num_samples:int=100, use_all:bool=False, plot_directly:bool=False,
                      save_path:Optional[str]=None):
    # check
    assert x.dim() == 2 and y.dim() <= 2
    assert x.size(0) == y.size(0)
    assert x.device == y.device
    if mask is not None:
        assert mask.size(0) == x.size(0)
    num_nodes = x.size(0)
    if y.dim() == 2: # one-hot label
        y = torch.argmax(y, dim=1)
    
    if use_all:
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
    else:
        if mask is None:
            mask = torch.randperm(num_nodes)[:num_samples]
        x = x[mask].detach().numpy()
        y = y[mask].detach().numpy()

    x_embed = TSNE(n_components=2, perplexity=50).fit_transform(x)
    x_1, x_2 = map(lambda array:array.squeeze(), np.split(x_embed, 2, axis=1))

    fig, ax = plt.subplots(figsize=(10,10))
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.scatter(x_1, x_2, c=y, marker=".", cmap="Set1")

    if plot_directly:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def plot_spectral_figure(x:Tensor, edge_index:Tensor, edge_weight:Optional[Tensor]=None,
                         ylim:float=0.1, save_path:Optional[str]=None):
    # sparse to dense
    num_nodes = x.size(0)
    assert x.dim() == 1
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float, device=edge_index.device)
    size = torch.Size([num_nodes, num_nodes] + list(edge_weight.size())[1:])
    edge_index, edge_weight = get_laplacian(edge_index=edge_index, edge_weight=edge_weight,
                                            normalization="sym", num_nodes=num_nodes)
    laplacian = torch.sparse_coo_tensor(edge_index, values=edge_weight, size=size)
    laplacian = laplacian.to_dense()

    # Computes the eigenvalues and eigenvectors
    evalues, evectors = torch.eig(laplacian, eigenvectors=True)
    evalues = evalues[:, 0].detach().cpu().numpy() # only real part
    spectral_signal = torch.matmul(evectors.T, x).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10,10))
    plt.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05, hspace=0, wspace=0)
    markerline, stemlines, baseline = ax.stem(evalues, spectral_signal, '-', markerfmt=",", linefmt="blue")
    plt.setp(baseline, color='black', linewidth=2)
    plt.ylim(-ylim, ylim)
    plt.savefig(save_path)


def plot_xs(xs:Dict[str,Tensor], save_path:Optional[str]=None):
    fig, ax = plt.subplots(figsize=(10,10))
    def plot_x(_x:Tensor, name:str):
        nid = np.arange(_x.size(0))
        _x = _x.detach().numpy()
        ax.plot(nid, _x, label=name)

    for k, v in xs.items():
        plot_x(v, k)

    plt.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05, hspace=0, wspace=0)
    plt.legend()
    plt.savefig(save_path)


def plot_curves(xs:Dict[str, list], save_path:Optional[str]=None):
    _, ax = plt.subplots(figsize=(10, 10))
    def plot_x(_x:list, name:str):
        nid = np.arange(len(_x))
        ax.plot(nid, _x, label=name)
    
    for k, v in xs.items():
        plot_x(v, k)
    
    plt.subplots_adjust(top=0.95, bottom=0.075, right=0.95, left=0.13, hspace=0, wspace=0)
    plt.ylim(0.8, 1)

    plt.legend(fontsize=25)
    plt.xlabel("Epoch", fontsize=25)
    plt.ylabel("Accuracy", fontsize=25)
    plt.tick_params(labelsize=20)
    plt.savefig(save_path)
