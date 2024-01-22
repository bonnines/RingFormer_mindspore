import inspect
from torch.nn import Sequential,  ReLU
from torch.nn import MultiheadAttention
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.aggr.utils import (
    MultiheadAttentionBlock,
    SetAttentionBlock,
)
from torch_geometric.nn import SimpleConv, HeteroConv, GINEConv, SAGEConv, GATConv, GINConv,  GCNConv, GPSConv, Linear, global_add_pool,global_mean_pool, global_max_pool, dense_diff_pool, DenseGINConv
from torch_geometric.nn import TopKPooling,  SAGPooling

import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch, to_dense_adj, add_self_loops, add_remaining_self_loops
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from copy import deepcopy

from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor, OptTensor
from torch_geometric.utils import spmm
from torch_geometric.nn.inits import reset



import inspect
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential


from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax


import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential,  ReLU
from torch_scatter import scatter
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor, OptTensor
from torch_geometric.typing import Adj
from torch import Tensor




class SparseEdgeConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        bias: bool = True,
        root_weight: bool = True,
        combine: str = 'add',
        clip_attn: bool = False, 
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.combine = combine
        self.clip_attn = clip_attn
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)

        if self.combine.startswith('cat'):
            if self.combine[-1] == '1':
                self.lin_combine = Sequential(Linear(in_channels[0]*2, in_channels[0]))
            elif self.combine[-1] == '2':
                self.lin_combine = Sequential(Linear(in_channels[0]*2, in_channels[0]), nn.Dropout(dropout))
            else:
                self.lin_combine = Sequential(Linear(in_channels[0]*2, in_channels[0]), ReLU())
        elif self.combine.startswith('add_lin'):
            self.lin_combine = Sequential(Linear(in_channels[0], in_channels[0]), ReLU())
        elif self.combine.startswith('lin_add'):
            self.lin_combine = Linear(in_channels[0], in_channels[0])
        elif self.combine.startswith('dual_lin_add'):
            self.lin_combine0 = Linear(in_channels[0], in_channels[0])
            self.lin_combine1 = Linear(in_channels[0], in_channels[0])
        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)



        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=x[1], key=x[0], value=x[0],
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            out = out + x_r
        else:
            out = out + x[1]

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        assert edge_attr is not None
        
        H, C = self.heads, self.out_channels
        
        if self.combine == 'add':
            key_j = value_j = key_j + edge_attr
            #  value_j = value_j + edge_attr
        elif self.combine.startswith('cat'):
            key_j = value_j = self.lin_combine(torch.cat([key_j, edge_attr], dim=-1))
            # value_j = self.lin_combine(torch.cat([value_j, edge_attr], dim=-1))
        elif self.combine == 'add_lin':
            key_j = value_j = self.lin_combine(key_j + edge_attr)
            # value_j = self.lin_combine(value_j + edge_attr)
        elif self.combine == 'lin_add':
            edge_attr = self.lin_combine(edge_attr)
            key_j = value_j = (key_j + edge_attr).relu()
            # value_j = (value_j + edge_attr).relu()  
        elif self.combine.startswith('dual_lin_add'):
            if self.combine[-1] == '1':
                edge_attr = self.lin_combine0(edge_attr)
                key_j, value_j = self.lin_combine0(key_j), self.lin_combine0(value_j)
                key_j = (key_j + edge_attr).relu()
                value_j = (value_j + edge_attr).relu()
            elif self.combine[-1] == '2':
                edge_attr = self.lin_combine0(edge_attr)
                key_j, value_j = self.lin_combine0(key_j), self.lin_combine0(value_j)
                key_j = key_j + edge_attr
                value_j = value_j + edge_attr
            elif self.combine[-1] == '3':
                edge_attr = self.lin_combine0(edge_attr)
                key_j, value_j = self.lin_combine0(key_j), self.lin_combine0(value_j)
                key_j = F.dropout(key_j + edge_attr, p=self.dropout, training=self.training)
                value_j = F.dropout(value_j + edge_attr, p=self.dropout, training=self.training) 
            elif self.combine[-1] == '4':
                edge_attr = self.lin_combine0(edge_attr)
                key_j = value_j = self.lin_combine1(key_j) + edge_attr
                # key_j = key_j 
                # value_j = value_j + edge_attr      
            elif self.combine[-1] == '5':
                edge_attr = self.lin_combine0(edge_attr)
                key_j = value_j = (self.lin_combine1(key_j) + edge_attr).relu()                                  
        else:
            raise NotImplementedError
        
        query_i = self.lin_query(query_i).view(-1, H, C)
        key_j = self.lin_key(key_j).view(-1, H, C)
        value_j = self.lin_value(value_j).view(-1, H, C)

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        if self.clip_attn:
            alpha = alpha.clamp(-5, 5)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
        

def get_activation(activation):
    if activation == 'relu':
        return 2, nn.ReLU()
    elif activation == 'gelu':
        return 2, nn.GELU()
    elif activation == 'silu':
        return 2, nn.SiLU()
    elif activation == 'glu':
        return 1, nn.GLU()
    else:
        raise ValueError(f'activation function {activation} is not valid!')
            
        
class SparseEdgeFullLayer(nn.Module):
    """Exphormer attention + FFN
    """

    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0,
                 dim_edge=None,
                 layer_norm=True,
                 activation = 'relu',
                 root_weight=True,
                 residual=True, use_bias=False, combine='add',
                 clip_attn=False,
                 **kwargs):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm

        self.attention = SparseEdgeConv(in_dim, out_dim//num_heads, heads=num_heads, root_weight=root_weight, 
                                        dropout=dropout, concat=True, edge_dim=dim_edge, use_bias=use_bias, 
                                        combine=combine, clip_attn=clip_attn)

        self.O_h = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)



        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        factor, self.activation_fn = get_activation(activation=activation)
        self.FFN_h_layer2 = nn.Linear(out_dim * factor, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)


    #     self.reset_parameters()

    # def reset_parameters(self):
    #     xavier_uniform_(self.attention.Q.weight, gain=1 / math.sqrt(2))
    #     xavier_uniform_(self.attention.K.weight, gain=1 / math.sqrt(2))
    #     xavier_uniform_(self.attention.V.weight, gain=1 / math.sqrt(2))
    #     xavier_uniform_(self.attention.E.weight, gain=1 / math.sqrt(2))
    #     xavier_uniform_(self.O_h.weight, gain=1 / math.sqrt(2))
    #     constant_(self.O_h.bias, 0.0)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr:Tensor, **kwargs):
        h = x
        h_in1 = h  # for first residual connection

        # multi-head attention out
        h = self.attention(x, edge_index, edge_attr)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)


        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = self.activation_fn(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        h = h
        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual)
        
        
        
from torch_geometric.nn import TransformerConv as UMPConv
class Het_Transfomer(torch.nn.Module):
    def __init__(self, metadata, dim, num_gc_layers, gnn='GINE', inter_gnn='GINE', ring_gnn='GPS', norm=None, transformer_norm=None, aggr='sum', jk='cat', 
                 dropout = 0.0, attn_dropout=0.0, pool = 'add', first_residual = False, residual=False, heads=4, use_bias=False,
                 padding=True, init_embs=False, mask_non_edge = False, add_mol=False, combine_mol = 'add', root_weight=True, 
                 combine_edge='add', clip_attn=False, **kwargs):
        super(Het_Transfomer, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.jk = jk
        self.dropout = dropout
        self.norms = None
        self.residual = residual
        self.first_residual = first_residual
        self.aggr = aggr
        self.use_edge_attr = True
        self.ring_gnn = ring_gnn
        self.add_mol = add_mol
        self.combine_mol = combine_mol
        
        assert norm is None
        
        if 'mol' in metadata[0]:
            self.use_mol = True
            print('Adding Mol node to heterogenous graph!')
        else:
            self.use_mol = False
        if 'pair' in metadata[0]:
            self.use_pair = True
            print('Adding Pair node to heterogenous graph!')
        else:
            self.use_pair = False            
                                
        if pool == 'add':
            self.pool = global_add_pool
        elif pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool        
        if norm is not None:
            norm_layer = normalization_resolver(
                norm,
                dim,
            )
            self.norms = torch.nn.ModuleList()
        if 'cat' in aggr:
            self.lin_atom = torch.nn.ModuleList()
            self.lin_ring = torch.nn.ModuleList()
            if self.use_mol:
                self.lin_mol = torch.nn.ModuleList()
            if self.use_pair:
                self.lin_pair = torch.nn.ModuleList()
        if gnn == 'GIN':
            nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            gnn_conv = GINConv(nn)
        elif gnn == 'GINE':
            nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            gnn_conv = GINEConvV2(nn) 
        elif gnn == 'GPS':
            nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            gnn_conv = GPSConv(dim, GINEConvV2(nn), heads=heads, norm = transformer_norm,
                           attn_dropout=attn_dropout, dropout=dropout) 
        elif gnn == 'GAT':
            gnn_conv = GATConvV2(dim, dim,  edge_dim=dim, heads=heads, dropout=dropout,  concat=False, add_self_loops=False) 
        elif gnn == 'SAGE':
            gnn_conv = SAGEConv_edgeattr(dim, dim, normalize=False, aggr='mean')                                           
        elif gnn == 'Simple':
            gnn_conv = SimpleConv()
            self.use_edge_attr = False
        elif gnn == 'Gated':
            gnn_conv =  ResGatedGraphConv(dim, dim, edge_dim=dim)
        else:
            raise NotImplementedError
        
        if inter_gnn == gnn:
            inter_gnn_conv = gnn_conv
        elif inter_gnn == 'GINE':
            nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            inter_gnn_conv = GINEConvV2(nn)             
        elif inter_gnn == 'GAT':
            inter_gnn_conv = GATConv(dim, dim,  heads=heads, dropout=dropout,  concat=False, add_self_loops=False)
        elif inter_gnn == 'SAGE':
            inter_gnn_conv = SAGEConv_edgeattr(dim, dim, normalize=False, aggr='mean')            
        elif inter_gnn == 'SAGE_add':
            inter_gnn_conv = SAGEConv_edgeattr(dim, dim, normalize=False, aggr='add')        
        else:
            raise NotImplementedError
        
        if 'GINE' in ring_gnn:            
            ring_gnn_conv = gnn_conv
        elif ring_gnn == 'GPS':
            nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            ring_gnn_conv = GPSConv(dim, GINEConvV2(nn), heads=heads, norm = transformer_norm,
                           attn_dropout=attn_dropout, dropout=dropout)      
        elif ring_gnn == 'Transformer':
            ring_gnn_conv = GPSConv(dim, None, heads=heads, norm = transformer_norm,
                                    attn_dropout=attn_dropout, dropout=dropout)   
        elif ring_gnn == 'UMPConv':
            ring_gnn_conv = UMPConv(dim, dim, concat=False, edge_dim=dim)
        elif ring_gnn == 'Graphormer':
            ring_gnn_conv = GraphormerEncoderLayer(node_dim=dim, edge_dim=dim, n_heads=heads, max_path_distance=30)
            
            self.centrality_encoding = CentralityEncoding(
            max_in_degree=10,
            max_out_degree=10,
            node_dim=dim
        )

            self.spatial_encoding = SpatialEncoding(
                max_path_distance=30,
            )
        elif ring_gnn == 'TransformerConv':
            ring_gnn_conv = TransformerConv(dim, heads=heads, norm = transformer_norm,
                           dropout=dropout, padding=padding, init_embs=init_embs, mask_non_edge=mask_non_edge)
        elif ring_gnn == 'Exphormer':
            ring_gnn_conv = ExphormerFullLayer(dim, dim, num_heads=heads, dropout=dropout, dim_edge=dim, residual=residual, use_bias=use_bias)
        elif ring_gnn == 'UniMP':
            ring_gnn_conv = UniMPFullLayer(dim, dim, num_heads=heads, dropout=dropout, dim_edge=dim, root_weight=root_weight, residual=residual, use_bias=use_bias)
        elif ring_gnn == 'SparseEdge':
            ring_gnn_conv = SparseEdgeFullLayer(dim, dim, num_heads=heads, dropout=dropout, dim_edge=dim, 
                                                residual=residual, use_bias=use_bias, combine=combine_edge, 
                                                root_weight=root_weight, clip_attn=clip_attn)
        else:
            raise NotImplementedError
        
        num_atom_messages = 0
        num_ring_messages = 0
        num_pair_messages = 0
        for rel in metadata[1]:
            if rel[-1] == 'atom':
                num_atom_messages += 1
            elif rel[-1] == 'ring':
                num_ring_messages += 1
            elif rel[-1] == 'pair':
                num_pair_messages += 1
        for _ in range(num_gc_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                if edge_type[0] == edge_type[-1]: # intra
                    if edge_type[-1] == 'ring':
                        conv_dict[edge_type] = deepcopy(ring_gnn_conv)
                    elif edge_type[-1] == 'atom':
                        conv_dict[edge_type] = deepcopy(gnn_conv)
                    else:
                        raise NotImplementedError
                else:
                    conv_dict[edge_type] = deepcopy(inter_gnn_conv)
            conv = HeteroConv(conv_dict, aggr='cat' if 'cat' in aggr else aggr)
            self.convs.append(conv)
            if aggr == 'cat':
                self.lin_atom.append(Sequential(Linear(num_atom_messages*dim, dim), ReLU()))
                self.lin_ring.append(Sequential(Linear(num_ring_messages*dim, dim), ReLU()))
                if self.use_mol:
                    self.lin_mol.append(Sequential(Linear(dim, dim), ReLU()))
                if self.use_pair:
                    self.lin_pair.append(Sequential(Linear(dim*num_pair_messages, dim), ReLU()))
                
            elif aggr == 'cat_self':
                self.lin_atom.append(Sequential(Linear((num_atom_messages+1)*dim, dim), ReLU()))
                self.lin_ring.append(Sequential(Linear((num_ring_messages+1)*dim, dim), ReLU()))
                if self.use_mol:
                    self.lin_mol.append(Sequential(Linear(2*dim, dim), ReLU()))
                if self.use_pair:
                    self.lin_pair.append(Sequential(Linear(2*dim, dim), ReLU()))
                    
    def forward(self, x_dict, edge_index_dict, batch_dict, edge_attr_dict = None,  edge_type_dict=None, data = None,):
        x_atom = [x_dict['atom']] if self.first_residual else []
        x_ring = [x_dict['ring']] if self.first_residual else []
        # if self.use_mol:
        #     x_mol = [x_dict['mol']] if self.first_residual else []
        # if self.use_pair:
        #     x_pair = [x_dict['pair']] if self.first_residual else []
            
            
        # Graphormer pre-processing

        b_dict = {('ring','r2r','ring'): None}
        edge_paths_dict = {('ring','r2r','ring'): None}
        ptr_dict = {('ring','r2r','ring'): None}
        # Convolution
        for i, conv in enumerate(self.convs):
            if self.use_edge_attr:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict, batch_dict=batch_dict, 
                              b_dict=b_dict, edge_paths_dict=edge_paths_dict, ptr_dict=ptr_dict, edge_type_dict=edge_type_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict, batch_dict=batch_dict, 
                              b_dict=b_dict, edge_paths_dict=edge_paths_dict, ptr_dict=ptr_dict, edge_type_dict=edge_type_dict)
            x_dict = {key: F.dropout(F.relu(x), p=self.dropout, training=self.training) for key, x in x_dict.items()}
            if self.aggr == 'cat':
                x_dict['atom'] = F.dropout(self.lin_atom[i](x_dict['atom']), p=self.dropout, training=self.training)
                x_dict['ring'] = F.dropout(self.lin_ring[i](x_dict['ring']), p=self.dropout, training=self.training)
                # if self.use_mol:
                #     x_dict['mol'] = F.dropout(self.lin_mol[i](x_dict['mol']), p=self.dropout, training=self.training)
                # if self.use_pair:
                #     x_dict['pair'] = F.dropout(self.lin_pair[i](x_dict['pair']), p=self.dropout, training=self.training)
            elif self.aggr == 'cat_self':
                x_dict['atom'] = F.dropout(self.lin_atom[i](torch.cat((x_atom[-1], x_dict['atom']), -1)), p=self.dropout, training=self.training)
                x_dict['ring'] = F.dropout(self.lin_ring[i](torch.cat((x_ring[-1], x_dict['ring']), -1)), p=self.dropout, training=self.training)
                # if self.use_mol:
                #     x_dict['mol'] = F.dropout(self.lin_mol[i](torch.cat((x_mol[-1], x_dict['mol']), -1)), p=self.dropout, training=self.training)
                # if self.use_pair:
                #     x_dict['pair'] = F.dropout(self.lin_pair[i](torch.cat((x_pair[-1], x_dict['pair']), -1)), p=self.dropout, training=self.training)
            x_atom.append(x_dict['atom'])
            x_ring.append(x_dict['ring'])
            # if self.use_mol:
            #     x_mol.append(x_dict['mol'])
            # if self.use_pair:
            #     x_pair.append(x_dict['pair'])
            
        if self.jk == 'cat':
            x_atom = torch.cat(x_atom, 1)
            x_ring = torch.cat(x_ring, 1)
            # if self.use_mol:
            #     x_mol = torch.cat(x_mol, 1)
            # if self.use_pair:
            #     x_pair = torch.cat(x_pair, 1)
        elif self.jk == 'last':
            x_atom = x_atom[-1]
            x_ring = x_ring[-1]
            # if self.use_mol:
            #     x_mol = x_mol[-1]
            # if self.use_pair:
            #     x_pair = x_pair[-1]
                
        x_atom = self.pool(x_atom, batch_dict['atom'])
        if self.add_mol:
            x_ring_out = self.pool(x_ring[data['ring'].ring_mask], batch_dict['ring'][data['ring'].ring_mask])
            x_mol = self.pool(x_ring[~data['ring'].ring_mask], batch_dict['ring'][~data['ring'].ring_mask])
            if self.combine_mol == 'add':
                x_ring_out = x_ring_out + x_mol
            elif self.combine_mol == 'cat':
                x_ring_out = torch.cat((x_ring_out, x_mol), -1)
            elif self.combine_mol == 'drop':
                x_ring_out = x_ring_out
            else:
                raise NotImplementedError
        else:
            x_ring_out = self.pool(x_ring, batch_dict['ring'])
            x_mol = None
            
            
        # if self.use_pair:
        #     x_pair = self.pool(x_pair, batch_dict['pair'])
        # else:
        #     x_pair = None
        # if not self.use_mol:
        #     x_mol = None
        return x_atom, x_ring_out, None, x_mol