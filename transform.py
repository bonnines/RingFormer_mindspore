from rdkit import Chem
from rdkit.Chem import rdDepictor
from collections import defaultdict
from math import atan2

from typing import Any, Optional

import numpy as np
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    scatter,
    to_edge_index,
    to_scipy_sparse_matrix,
    to_torch_csr_tensor,
    degree,
    one_hot,
)
import random
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 
import networkx as nx



class IndexGraph(object):
    def __init__(self,  start = 0):
        self.current = start
    def __call__(self, data):
        data['graph_id'] = torch.LongTensor([self.current]).reshape(1,-1)
        self.current += 1
        return data

class MaskAtom(object):
    def __init__(self,  mask_rate=0.15, seed=None):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.full_atom_feature_dims = get_atom_feature_dims()
        self.mask_rate = mask_rate
        self.seed = seed
        super(MaskAtom, self).__init__()
        
    def __call__(self, data, masked_atom_indices=None):
        """
        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            if self.seed is not None:
                np.random.seed(self.seed)
                random.seed(self.seed)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        masked_atom_indices = torch.tensor(masked_atom_indices)

        # modify the original node feature of the masked node
        x = data.x.clone()
        for atom_idx in masked_atom_indices:
            x[atom_idx] = torch.tensor(self.full_atom_feature_dims).long()
        new_data = Data(x=x, edge_attr=data.edge_attr, edge_index=data.edge_index, y=data.y, mask_node_label=mask_node_label, masked_atom_indices=masked_atom_indices)
        return new_data
    
    
# class AddRingRandomWalkPE(object):
#     r"""Adds the random walk positional encoding from the `"Graph Neural
#     Networks with Learnable Structural and Positional Representations"
#     <https://arxiv.org/abs/2110.07875>`_ paper to the given graph
#     (functional name: :obj:`add_random_walk_pe`).

#     Args:
#         walk_length (int): The number of random walk steps.
#         attr_name (str, optional): The attribute name of the data object to add
#             positional encodings to. If set to :obj:`None`, will be
#             concatenated to :obj:`data.x`.
#             (default: :obj:`"random_walk_pe"`)
#     """
#     def __init__(
#         self,
#         walk_length: int,
#         attr_name: Optional[str] = 'ring_pe',
#     ):
#         self.walk_length = walk_length
#         self.attr_name = attr_name

#     def __call__(self, data: Data) -> Data:
#         row, col = data.ring_edge_index
#         N = data.num_rings.item()



#         value = torch.ones(data.num_re.item(), device=row.device)
#         value = scatter(value, row, dim_size=N, reduce='sum').clamp(min=1)[row]
#         value = 1.0 / value

#         adj = to_torch_csr_tensor(data.ring_edge_index, value)

#         out = adj
#         pe_list = [get_self_loop_attr(*to_edge_index(out), num_nodes=N)]
#         for _ in range(self.walk_length - 1):
#             out = out @ adj
#             pe_list.append(get_self_loop_attr(*to_edge_index(out), N))
#         pe = torch.stack(pe_list, dim=-1)

#         data[self.attr_name] = pe
#         return data
    
    
class AddRingRandomWalkPE(object):
    r"""Adds the random walk positional encoding from the `"Graph Neural
    Networks with Learnable Structural and Positional Representations"
    <https://arxiv.org/abs/2110.07875>`_ paper to the given graph
    (functional name: :obj:`add_random_walk_pe`).

    Args:
        walk_length (int): The number of random walk steps.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"random_walk_pe"`)
    """
    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = 'ring_pe',
    ):
        self.walk_length = walk_length
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        edge_index = data[('ring','r2r','ring')].edge_index
        row, col = edge_index
        N = data['ring'].num_nodes



        value = torch.ones(data[('ring','r2r','ring')].edge_index.shape[1], device=row.device)
        value = scatter(value, row, dim_size=N, reduce='sum').clamp(min=1)[row]
        value = 1.0 / value

        adj = to_torch_csr_tensor(edge_index, value)

        out = adj
        pe_list = [get_self_loop_attr(*to_edge_index(out), num_nodes=N)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            pe_list.append(get_self_loop_attr(*to_edge_index(out), N))
        pe = torch.stack(pe_list, dim=-1)

        data['ring'][self.attr_name] = pe
        return data
    
class AddRingDegreePE(object):
    def __init__(
        self,
        max_degree: int,
        attr_name: Optional[str] = 'ring_pe',
    ):
        self.max_degree = max_degree
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        row, col = data.ring_edge_index
        N = data.num_rings.item()
        deg = degree(row, num_nodes=N, dtype=torch.long)
        deg = one_hot(deg, num_classes=self.max_degree + 1)
        data[self.attr_name] = deg
        return data

class AddHetRingDegreePE(object):
    def __init__(
        self,
        max_degree: int,
        attr_name: Optional[str] = 'ring_pe',
        one_hot: bool = False,
    ):
        self.max_degree = max_degree
        self.attr_name = attr_name
        self.one_hot = one_hot

    def __call__(self, data: Data) -> Data:
        row, col = data[('ring','r2r','ring')].edge_index
        N = data['ring'].num_nodes
        deg = degree(row, num_nodes=N, dtype=torch.long)
        
        if self.one_hot:
            deg = one_hot(deg, num_classes=self.max_degree + 1)
            data['ring'][self.attr_name] = deg
        else:
            data['ring'][self.attr_name] = deg.int()
        return data 

class AddHetRingTypeDegreePE(object):
    def __init__(
        self,
        attr_name: Optional[str] = 'ring_pe',
        num_edge_type: int = 17,
    ):
        self.attr_name = attr_name
        self.num_edge_type = num_edge_type

    def __call__(self, data: Data) -> Data:
        row, col = data[('ring','r2r','ring')].edge_index
        edge_attr = data[('ring','r2r','ring')].edge_attr
        N = data['ring'].num_nodes
        degs = []
        for type in range(self.num_edge_type):
            edge_mask = edge_attr == type
            deg = degree(row[edge_mask], num_nodes=N, dtype=torch.long).reshape(-1,1)
            degs.append(deg)
        degs = torch.cat(degs, dim=1)
        data['ring'][self.attr_name] = degs.int()
        return data 


def edge_index_to_networkx(edge_index):
    G = nx.Graph()
    G.add_edges_from(edge_index.T.numpy())
    return G

class AddHetRingPathPE(object):
    def __init__(
        self,
        max_length: int = 30,
        attr_name: Optional[str] = 'ring_pe',
        method = 'absolute',
    ):
        self.max_length = max_length
        self.attr_name = attr_name
        self.method = method

    def __call__(self, data: Data) -> Data:
        edge_index = data[('ring','r2r','ring')].edge_index
        g = edge_index_to_networkx(edge_index)
        max_sp = []
        for n in g.nodes:
            length = list(nx.shortest_path_length(g, source=n).values())
            max_sp.append(max(length))
        if self.method == 'absolute':
            max_sp = torch.as_tensor(max_sp)
        elif self.method == 'relative':
            max_sp = torch.as_tensor(max_sp) - min(max_sp)
        else:
            raise ValueError('method must be absolute or relative')
        max_sp = one_hot(max_sp, num_classes=self.max_length + 1)
        data['ring'][self.attr_name] = max_sp
        return data 

class DelHetEdgeAttr(object):
    def __init__(
        self,
        targe_edge_type = ('ring', 'r2r', 'ring'),
    ):

        self.targe_edge_type = targe_edge_type

    def __call__(self, data: Data) -> Data:
        data[self.targe_edge_type].edge_attr = torch.zeros_like(data[self.targe_edge_type].edge_attr, dtype=torch.long)
        return data

class DelHetEdgeType(object):
    def __init__(
        self,
        targe_edge_type = ('ring', 'r2a', 'atom'),
    ):

        self.targe_edge_type = targe_edge_type

    def __call__(self, data: Data) -> Data:
        new_data = HeteroData()
        for key, value in data.stores[0].items():
            new_data[key] = value
        for node_type in data.node_types:
            for key, value in data[node_type].items():
                new_data[node_type][key] = value
        for edge_type in data.edge_types:
            if edge_type == self.targe_edge_type:
                continue
            for key, value in data[edge_type].items():
                new_data[edge_type][key] = value
        return new_data


class AddVirtualMol(object):
    
    """Add virtual mol node to the ring graph, use ring_mask to indicate the virtual mol node
    """
    def __init__(
        self,
        version: str = 'V1', # lower node type name
        num_ring_types = 58, 
    ):
        self.version = version
        if version == 'V1':
            virtual_edge_attr = 22
        elif version == 'V2':
            virtual_edge_attr = 41
        elif version == 'V3':
            virtual_edge_attr = 18
        elif version == 'V4':
            virtual_edge_attr = 5
        elif version == 'V5':
            virtual_edge_attr = 7        
        elif version == 'V6':
            virtual_edge_attr = 12           
        elif version == 'V7':
            virtual_edge_attr = 11 
        elif version == 'V8':
            virtual_edge_attr = 12   
        elif version == 'BRICS':
            virtual_edge_attr = 300
        else:
            raise NotImplementedError
        self.virtual_edge_attr = virtual_edge_attr
        self.num_ring_types = num_ring_types
    def __call__(self, data: HeteroData) -> HeteroData:
        x, edge_index, edge_attr = data['ring'].x, data[('ring', 'r2r', 'ring')].edge_index, data[('ring', 'r2r', 'ring')].edge_attr
        num_nodes = data['ring'].num_nodes
        # node 
        x = torch.cat([x, torch.LongTensor([self.num_ring_types+1]).reshape(1,1)], dim=0)
        ring_mask = torch.ones((num_nodes+1, ), dtype=torch.bool)
        ring_mask[-1] = False
        if hasattr(data['ring'], 'ring_pe'):
            if len(data['ring'].ring_pe.shape) == 1:
                data['ring'].ring_pe = torch.cat((data['ring'].ring_pe, torch.zeros((1,), dtype=data['ring'].ring_pe.dtype)), dim=0)        
            else:
                data['ring'].ring_pe = torch.cat((data['ring'].ring_pe, torch.zeros((1, data['ring'].ring_pe.shape[1]), dtype=data['ring'].ring_pe.dtype)), dim=0)
        # edge index
        virtual_src = torch.cat((torch.arange(num_nodes), torch.full((num_nodes, ), num_nodes)), dim=0)
        virtual_dst = torch.cat((torch.full((num_nodes, ), num_nodes), torch.arange(num_nodes)), dim=0)
        virtual_edge_index = torch.stack((virtual_src, virtual_dst), dim=0)
        edge_index = torch.cat((edge_index, virtual_edge_index), dim=1)
        # edge attr
        if self.version != 'BRICS':
            edge_attr = torch.cat((edge_attr, torch.full((virtual_edge_index.size(1),), self.virtual_edge_attr)), dim=0)
        else:
            edge_attr = torch.cat((edge_attr, torch.Tensor([23, 7, 3]).repeat(virtual_edge_index.size(1),1).long()), dim=0)



        data['ring'].x = x
        data['ring'].ring_mask = ring_mask
        data[('ring', 'r2r', 'ring')].edge_index = edge_index
        data[('ring', 'r2r', 'ring')].edge_attr = edge_attr
        return data

class RingTypeConvertor(object):
    def __init__(
        self,
        # threshold = 50,
        dataset = 'CEPDB'
    ):
        if dataset == 'CEPDB':
            self.convetor = torch.load(f'/data/CEPDB/BRICS/conversion_50.pt')
        elif dataset == 'HOPV':
            self.convetor = torch.load(f'./data/HOPV/conversion_5.pt')
        elif dataset == 'PolymerFA':
            self.convetor = torch.load(f'./data/Polymer_FA/conversion_5.pt')
        elif dataset == 'pNFA':
            self.convetor = torch.load(f'./data/Polymer_NFA_p/conversion_5.pt')
        elif dataset == 'nNFA':
            self.convetor = torch.load(f'./data/Polymer_NFA_n/conversion_5.pt')
        else:
            raise NotImplementedError

    def __call__(self, data: Data) -> Data:
        data['ring'].x = self.convetor[data['ring'].x]

        return data

class AddMolNode(object):
    def __init__(
        self,
        name: str = 'ring', # lower node type name
    ):
        self.name = name  
    def __call__(self, data: HeteroData) -> HeteroData:
        data['mol'].x= torch.LongTensor([0])
        dst = torch.LongTensor([i for i in range(data[self.name].num_nodes)])
        src = torch.zeros_like(dst).long()
        num_edges = len(dst)
        data['mol', f'm2{self.name[0]}', self.name].edge_index = torch.vstack([src, dst])
        data['mol', f'm2{self.name[0]}', self.name].edge_attr = torch.zeros((num_edges, 1)).long().reshape(-1)  
        
        data[self.name, f'{self.name[0]}2m', 'mol'].edge_index = torch.vstack([dst, src])
        data[self.name, f'{self.name[0]}2m', 'mol'].edge_attr = torch.zeros((num_edges, 1)).long().reshape(-1)  
        return data



    
class AddPairNode(object):
    def __call__(self, data: HeteroData) -> HeteroData:
        ring_pairs = data[('ring', 'r2r', 'ring')].edge_index[:,::2].T
        data['pair'].x= torch.zeros((len(ring_pairs), ), dtype=torch.long)
        dst = ring_pairs.reshape(-1)
        src = torch.LongTensor([[i,i] for i in range(len(ring_pairs))]).reshape(-1)
        num_edges = len(dst)
        data['pair', 'p2r', 'ring'].edge_index = torch.vstack([src, dst])
        data['pair', 'p2r', 'ring'].edge_attr = torch.zeros((num_edges, 1)).long().reshape(-1)  
        
        data['ring', 'r2p', 'pair'].edge_index = torch.vstack([dst, src])
        data['ring', 'r2p', 'pair'].edge_attr = torch.zeros((num_edges, 1)).long().reshape(-1) 
        return data
    

class DelAttribute(object):
    def __init__(
        self,
        name: str = 'edge_attr',
    ):


        self.name = name


    def __call__(
        self,
        data
    ) :

        del data[self.name]

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'