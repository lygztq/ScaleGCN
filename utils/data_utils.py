import torch
import logging
import typing
from scipy.stats import t
import math
from torch_geometric.data import Data
import torch_sparse as sp
from torch_geometric.nn import GCNConv
from torch import Tensor
from torch_geometric.typing import OptTensor

def get_stats(array, conf_interval=False, name=None, stdout=False, logout=False):
    """Compute mean and standard deviation from an numerical array
    
    Args:
        array (array like obj): The numerical array, this array can be 
            convert to :obj:`torch.Tensor`.
        conf_interval (bool, optional): If True, compute the confidence interval bound (95%)
            instead of the std value. (default: :obj:`False`)
        name (str, optional): The name of this numerical array, for log usage.
            (default: :obj:`None`)
        stdout (bool, optional): Whether to output result to the terminal. 
            (default: :obj:`False`)
        logout (bool, optional): Whether to output result via logging module.
            (default: :obj:`False`)
    """
    eps = 1e-9
    array = torch.tensor(array)
    std, mean = torch.std_mean(array)
    std = std.item()
    mean = mean.item()

    if conf_interval:
        n = array.size(0)
        se = std / (math.sqrt(n) + eps)
        t_value = t.ppf(0.975, df=n-1)

    center = mean
    if conf_interval:
        err_bound = t_value * se
    else:
        err_bound = std

    # log and print
    if name is None:
        name = "array {}".format(id(array))
    log = "{}: {:.4f}(+-{:.4f})".format(name, center, err_bound)
    if stdout:
        print(log)
    if logout:
        logging.info(log)

    return center, err_bound


def sparse_to_dense(shape, edge_index, value=None, dtype=None):
    """Convert a sparse matrix(COO format) to dense matrix

    Args:
        shape (Union[tuple, list, int, torch.Size]): The shape of this matrix. If
            an integer :math:`m` is given, the shape will be treated as :math:`m \\times m`
        edge_index (torch.Tensor): Edge index of this matrix in COO sparse representation.
        value (torch.Tensor, optional): Value of this matrix, if :obj:`None`, all value
            will be treated as 1. (default: :obj:`None`)
        dtype (torch.dtype, optional): Data type for edge weight, default is :obj:`float` 
            (default: :obj:`None`)
    """
    if isinstance(shape, int):
        shape = (shape, shape)
    elif isinstance(shape, torch.Size):
        shape = tuple(shape)
    assert isinstance(shape, (list, tuple))

    if value is None:
        value = 1
    dense = torch.zeros(shape, dtype=dtype)
    dense[edge_index[0], edge_index[1]] = value
    
    return dense


def dense_to_sparse(dense, need_value=True):
    """ Convert a dense matrix to COO sparse matrix """
    edge_idx = torch.nonzero(dense != 0).t()
    if not need_value:
        return (edge_idx, torch.ones(edge_idx.shape[-1]))
    value = dense[edge_idx[0], edge_idx[1]]
    return (edge_idx, value)


def dense_equal(mat1:torch.Tensor, mat2:torch.Tensor, err=1e-5):
    """Compare two dense matrix according to a given error threshold

    Args:
        mat1 (torch.Tensor): The first matrix
        mat2 (torch.Tensor): The second matrix
        err (float, optional): The error threshold. (default: 1e-5)
    """
    if len(mat1.shape) != 2 or len(mat2.shape) != 2:
        return False
    if mat1.shape != mat2.shape:
        return False

    return (torch.abs(mat1 - mat2) < err).all().item()


def generate_adj_matrix(num_nodes, symmetric=True, sparsity=0.5, connected=True):
    """Generate a adjacency matrix (in dense format) randomly.
    
    Args:
        num_nodes (int): The number of nodes in this graph.
        symmetric (bool, optional): If set :obj:`True`, will generate a 
            symmetric adjacency matrix. (default: :obj:`True`)
        sparsity (float, optional): The parameter that controls the sparsity
            of the generated graph. This value should in (0, 1]. (default: 0.5)
        connected (bool, optional): If set :obj:`True`, will generate a 
            connected graph. (default: :obj:`True`)
    """
    A = torch.rand((num_nodes, num_nodes), dtype=torch.float)
    if symmetric:
        A = 0.5 * (A + A.t())
    A = (A < (1 - sparsity)).long() * (1 - torch.eye(num_nodes))
    if connected:
        start = torch.randperm(num_nodes)
        end = torch.cat([start[1:], start.new_full(size=[1], fill_value=start[0].item())])
        A[start, end] = 1
        A[end, start] = 1
    return A


def add_self_loop_dense(A:torch.Tensor, force=False, fill_value=1):
    """Add self-loop to a dense adjacency
    
    Args:
        A (torch.Tensor): The input dense adjacency.
        force (bool, optional): If :obj:`True`, directly add :math:`I*fill_value` 
            to the given adjacency, else only add self-loop
            if the node does not have one. (default: :obj:`False`)
        fill_value (Union[int, float], optional): The fill value of self-loop
            (default: :obj:`1`)
    """
    num_nodes = A.size(0)

    if force:
        return A + fill_value * torch.eye(num_nodes, dtype=A.dtype)
    A[torch.eye(num_nodes, dtype=torch.bool)] = fill_value
    return A


def fake_data(num_nodes, num_features, sparsity=0.5, one_hot=False, 
              directed=False, weighted=False, with_label=False, label_dim=None,
              add_self_loop=False, device="cpu"):
    r"""Generate a fake graph data for test or debug.

    Args:
        num_nodes (int): The number of nodes in this graph.
        num_features (int): The dimension of node features.
        sparsity (float, optional): The parameter that controls the sparsity
            of the generated graph. This value should in (0, 1]. (default: 0.5)
        one_hot (bool, optional): If set :obj:`True`, generate one-hot node feature.
            (default: :obj:`False`)
        directed (bool, optional): If set :obj:`True`, generate asymmetric adjacency
            matrix. (default: :obj:`False`)
        weighted (bool, optional): If set :obj:`True`, generate edge with weight.
            (default: :obj:`False`)
        with_label (bool, optional): If set :obj:`True`, generate label. (default: :obj:`False`)
        label_dim (int, optional): If :obj:`with_label` is :obj:`True`, this argument must be given.
            The dimension of label. (default: :obj:`None`)
        add_self_loop (bool, optional): If set :obj:`True`, add self-loop to all nodes.
            (default: :obj:`False`)
        device (str, optional): The device where this graph data in. (default: :obj:`"cpu"`)
    """
    device = torch.device(device)
    x = torch.randn((num_nodes, num_features), device=device)
    if one_hot:
        x = (torch.max(x, axis=-1, keepdim=True)[0] == x).float()
    A = generate_adj_matrix(num_nodes, not directed, sparsity=sparsity)
    if add_self_loop:
        A += torch.eye(num_nodes)
    edge_idx = torch.nonzero(A == 1).t().contiguous()
    edge_idx = edge_idx.to(device)

    data_arguments = {}
    if weighted:
        data_arguments["edge_attr"] = torch.abs(torch.randn(edge_idx.shape[-1], device=device))
    if with_label:
        if label_dim is None:
            raise ValueError("The argument 'label_dim' must be given when generate data with label.")
        data_arguments["y"] = torch.randint(label_dim, size=[num_nodes], dtype=torch.int, device=device)
    return Data(x=x, edge_index=edge_idx, **data_arguments)


def remove_nodes(remove_ids:Tensor, num_nodes:int, 
                 edge_index:Tensor, edge_weight:OptTensor=None,
                 reorder_nodes:bool=False):
    """Remove some nodes from a graph.
    This function will return the subgraph without removed node and
    edges.

    Args:
        remove_ids (Union[torch.Tensor, list[int]]): Tensor of node ids to be removed.
        num_nodes (int): Number of nodes in origin graph.
        edge_index (torch.Tensor): Edge indexs.
        edge_weight (OptTensor, optional): Edge weights. (default: :obj:`None`)
        reorder_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be reordered to hold consecutive indices
            starting from zero. (default: :obj:`False`)
    """
    device = edge_index.device
    if not isinstance(remove_ids, Tensor):
        remove_ids = torch.tensor(remove_ids, device=device).long()
    remove_ids = remove_ids.unique()
    node_mask = edge_index.new_full((num_nodes, ), dtype=torch.bool, fill_value=True, 
                                    device=device)
    node_mask[remove_ids] = False
    row, col = edge_index[0], edge_index[1]
    edge_mask = node_mask[row] & node_mask[col]

    if reorder_nodes:
        num_node_remains = num_nodes - remove_ids.size(0)
        node_id_remains = torch.arange(num_nodes, device=device)[node_mask]
        mappings = torch.zeros(num_nodes, dtype=torch.long, device=device)
        mappings[node_id_remains] = torch.arange(num_node_remains, device=device)
        
    edge_index = edge_index[:, edge_mask]
    edge_weight = edge_weight[edge_mask] if edge_weight is not None else None
    
    if reorder_nodes:
        edge_index = mappings[edge_index]

    return edge_index, edge_weight
