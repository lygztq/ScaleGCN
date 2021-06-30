import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul_, sum

from .data_utils import add_self_loop_dense

Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul_(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul_(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def normalize_adj_symmetric(edge_index:torch.Tensor, num_nodes:int, edge_weight=None, 
                            improved=False, dtype=None):
    """Normalization by :math:`D^{-1/2} (A+I) D^{-1/2}` .
    
    Args:  
        edge_index (torch.Tensor): The edge_index tensor.
        num_nodes (int): The number of nodes in this graph.
        edge_weight (torch.Tensor, optional): The weight tensor of edges in this graph.
            If not given, the graph will be treated as a unweighted graph. (default: :obj:`None`)
        improved (bool, optional): If true, the function computes :math:`D^{-1/2} (A+2I) D^{-1/2}`
            (default: :obj:`False`)
        dtype (torch.dtype, optional): Data type for edge weight, default is :obj:`float` (default: :obj:`None`)
    """
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, 
                                 device=edge_index.device)
    
    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def _normalize_adj_symmetric_dense(adj, improved=False):
    """Normalization of dense adjacency by :math:`D^{-1/2} (A+I) D^{-1/2}`.

    Note: This function is only for test or debug.

    Args:
        adj (torch.Tensor): The input adjacency matrix.
        improved (bool, optional): If true, the function computes :math:`D^{-1/2} (A+2I) D^{-1/2}`
            (default: :obj:`False`)
    """
    fill_value = 1 if not improved else 2
    
    adj = add_self_loop_dense(adj, fill_value=fill_value)
    degree = torch.sum(adj, dim=-1)
    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[degree_inv_sqrt == float("inf")] = 0
    return degree_inv_sqrt * adj * degree_inv_sqrt.reshape(-1, 1)


def normalize_adj(edge_index:torch.Tensor, num_nodes:int, edge_weight=None, 
                  dtype=None):
    """Normalization by :math:`D^{-1} A` .

    Args:  
        edge_index (torch.Tensor): The edge_index tensor.
        num_nodes (int): The number of nodes in this graph.
        edge_weight (torch.Tensor, optional): The weight tensor of edges in this graph.
            If not given, the graph will be treated as a unweighted graph. (default: :obj:`None`)
        dtype (torch.dtype, optional): Data type for edge weight, default is :obj:`float` (default: :obj:`None`)
    """
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, 
                                 device=edge_index.device)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = 1 / deg
    deg_inv[deg_inv == float("inf")] = 0

    return edge_index, deg_inv[row] * edge_weight


def _normalize_adj_dense(adj):
    """Normalization of dense adjacency matrix by :math:`D^{-1} A` .

    Note: This function is only for test or debug.

    Args:
        adj (torch.Tensor): The input adjacency matrix.
    """
    degree = torch.sum(adj, dim=-1)
    degree_inv = 1 / degree
    degree_inv[degree_inv == float("inf")] = 0
    return degree_inv.reshape(-1, 1) * adj


def normalize_adj_diag_enhance(edge_index:torch.Tensor, num_nodes:int, diag_lambda:float=1.0,
                               edge_weight=None, improved=False, dtype=None):
    """Normalization by :math:`A'=D^{-1}(A+I), A'=A'+lambda*diag(A')` .
    
    Args:  
        edge_index (torch.Tensor): The edge_index tensor.
        num_nodes (int): The number of nodes in this graph.
        diag_lambda (float): The fill value on the diagonal of result adjacency matrix.
        edge_weight (torch.Tensor, optional): The weight tensor of edges in this graph.
            If not given, the graph will be treated as a unweighted graph. (default: :obj:`None`)
        improved (bool, optional): If true, the function computes :math:`A'=D^{-1}(A+2I)`
            (default: :obj:`False`)
        dtype (torch.dtype, optional): Data type for edge weight, default is :obj:`float` (default: :obj:`None`)
    """
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, 
                                 device=edge_index.device)
    
    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float("inf")] = 0
    edge_weight = deg_inv[row] * edge_weight
    
    self_loop_idx = row == col
    addition_term = torch.zeros_like(edge_weight, dtype=dtype, device=edge_index.device)
    addition_term[self_loop_idx] = diag_lambda * edge_weight[self_loop_idx]

    return edge_index, edge_weight + addition_term

def _normalize_adj_diag_enhance_dense(adj, diag_lambda:float, improved=False):
    """Normalization of dense adjacency by 
    :math:`A'=D^{-1}(A+I), A'=A'+lambda*diag(A')`.

    Note: This function is only for test or debug.

    Args:
        adj (torch.Tensor): The input adjacency matrix.
        diag_lambda (float): The fill value on the diagonal of result adjacency matrix.
        improved (bool, optional): If true, the function computes :math:`A'=D^{-1}(A+2I)`
            (default: :obj:`False`)
    """
    fill_value = 1 if not improved else 2
    
    adj = add_self_loop_dense(adj, fill_value=fill_value)
    degree = torch.sum(adj, dim=-1)
    degree_inv = 1 / degree
    degree_inv[degree_inv == float("inf")] = 0
    adj = adj * degree_inv.reshape(-1, 1)

    adj += (diag_lambda * adj.diag()).diag()

    return adj


def adj_normalize_transform(data:Data, norm_method="diag_enhance", **kwargs):
    """Transform a :obj:`torch_geometric.data.Data` via given
    normalization method

    Args:
        data (torch_geometric.data.Data): The data to be transformed.
        norm_method (str, optional): The normalization method. optionas are
            'diag_enhance', 'symmetric' and 'normal'. (default: 'diag_enhance')
        **kwargs (optional): Arguments used for normalization method.  
    """
    logger = logging.getLogger("trainlog")
    logger.info("Performing adjacency normalization ({})...".format(norm_method))
    if norm_method == "normal":
        edge_index, edge_weight = normalize_adj(
            data.edge_index, num_nodes=data.num_nodes, 
            edge_weight=data.edge_attr, **kwargs)
    elif norm_method == "diag_enhance":
        edge_index, edge_weight = normalize_adj_diag_enhance(
            data.edge_index, num_nodes=data.num_nodes,
            edge_weight=data.edge_attr, **kwargs)
    elif norm_method == "symmetric":
        edge_index, edge_weight = normalize_adj_symmetric(
            data.edge_index, num_nodes=data.num_nodes,
            edge_weight=data.edge_attr, **kwargs)
    else:
        raise ValueError("Unkonwn adjacency matrix normalization method: {}".format(norm_method))
    logger.info("Done!")
    data.edge_index = edge_index
    data.edge_attr = edge_weight
    return data
