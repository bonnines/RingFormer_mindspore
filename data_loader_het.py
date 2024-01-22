import os
import re
import os.path as osp
from scipy import sparse as sp
import torch
import numpy as np
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Constant
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_scipy_sparse_matrix, degree, from_networkx, add_self_loops
# from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.transforms import OneHotDegree
import torch
import torch.nn.functional as F

from torch_geometric.data import Data

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, from_smiles

import sys

import deepchem as dc


import torch
from torch_geometric.data import InMemoryDataset, download_url
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from torch_geometric.data import HeteroData

from data_loader import *


class HOPVHetDataset(InMemoryDataset):
    def __init__(self, root='./data/HOPV/Het/', transform=None, pre_transform=None, pre_filter=None, version='V1'):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        return ''
    @property
    def processed_file_names(self) -> str:
        return f'data_{self.version}.pt'
    def download(self):
        pass

    def process(self):
        # if self.version == 'V1':
        #     self_loop_id = 21
        # elif self.version == 'V2':
        #     self_loop_id = 40
        # elif self.version == 'V3':
        #     self_loop_id = 17
        # elif self.version == 'V4':
        #     self_loop_id = 4        
        # elif self.version == 'V5':
        #     self_loop_id = 6
        # elif self.version == 'V6':
        #     self_loop_id = 11      
        # elif self.version == 'V7':
        #     self_loop_id = 10    
        # elif self.version == 'V8':
        #     self_loop_id = 11                                  
        # else:
        #     raise ValueError('Invalid version')       
        # Read data into huge `Data` list.
        data_list = []
        dataset = HOPVDataset()
        ring_graphs = torch.load(f'./data/HOPV/ring_graphs_V1.pt')
        for idx, data in enumerate(dataset):
            het_data = HeteroData()
            
            het_data.y = data.y
            # Atom graph
            het_data['atom'].x = data.x
            het_data['atom', 'a2a', 'atom'].edge_index = data.edge_index
            het_data['atom', 'a2a', 'atom'].edge_attr = data.edge_attr
            # Ring graph
            het_data['ring'].x = ring_graphs[idx].x
            if ring_graphs[idx].edge_index.shape[1] == 0:
                het_data['ring', 'r2r', 'ring'].edge_index = torch.LongTensor([[0], [0]])
                het_data['ring', 'r2r', 'ring'].edge_attr = torch.LongTensor([[self_loop_id]]).reshape(-1) # 21 is the index of the self-loop edge type            
            else:
                het_data['ring', 'r2r', 'ring'].edge_index = ring_graphs[idx].edge_index.long()  
                het_data['ring', 'r2r', 'ring'].edge_attr = ring_graphs[idx].edge_attr.long().reshape(-1)      
            # Ring-atom graph
            r2a_edge_index = []
            a2r_edge_index = []
            for ring_id in range(het_data['ring'].num_nodes):
                target_atoms = ring_graphs[idx].ring2atom[ring_graphs[idx].ring2atom_batch==ring_id].tolist()
                r2a_edge_index.append(torch.LongTensor([[ring_id for _ in range(len(target_atoms))], 
                                                                               [atom_id for atom_id in target_atoms]]))
                a2r_edge_index.append(torch.LongTensor([ [atom_id for atom_id in target_atoms],
                                                                               [ring_id for _ in range(len(target_atoms))]]))     
            het_data['ring', 'r2a', 'atom'].edge_index = torch.cat(r2a_edge_index, dim=1)
            het_data['atom', 'a2r', 'ring'].edge_index = torch.cat(a2r_edge_index, dim=1)
            het_data['ring', 'r2a', 'atom'].edge_attr = torch.ones(het_data['ring', 'r2a', 'atom'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data['atom', 'a2r', 'ring'].edge_attr = torch.ones(het_data['atom', 'a2r', 'ring'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data.smiles = data.smiles
            
            
            het_data.ring_atoms = ring_graphs[idx].ring2atom
            het_data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            het_data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
            data_list.append(het_data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class PolymerFAHetDataset(InMemoryDataset):
    def __init__(self, root='./data/Polymer_FA/Het/', transform=None, pre_transform=None, pre_filter=None, version='V1'):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'raw/Polymer_FA.csv'

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.version}.pt'
    def download(self):
        pass

    def process(self):
        if self.version == 'V1':
            self_loop_id = 21
        elif self.version == 'V2':
            self_loop_id = 40
        elif self.version == 'V3':
            self_loop_id = 17
        elif self.version == 'V4':
            self_loop_id = 4        
        elif self.version == 'V5':
            self_loop_id = 6
        elif self.version == 'V6':
            self_loop_id = 11      
        elif self.version == 'V7':
            self_loop_id = 10    
        elif self.version == 'V8':
            self_loop_id = 11                                  
        else:
            raise ValueError('Invalid version')          
        # Read data into huge `Data` list.
        data_list = []
        dataset = PolymerFADataset()
        ring_graphs = torch.load(f'./data/Polymer_FA/ring_graphs_V1.pt')
        for idx, data in enumerate(dataset):
            het_data = HeteroData()
            
            het_data.y = data.y
            # Atom graph
            het_data['atom'].x = data.x
            het_data['atom', 'a2a', 'atom'].edge_index = data.edge_index
            het_data['atom', 'a2a', 'atom'].edge_attr = data.edge_attr
            # Ring graph
            het_data['ring'].x = ring_graphs[idx].x
            if ring_graphs[idx].edge_index.shape[1] == 0:
                het_data['ring', 'r2r', 'ring'].edge_index = torch.LongTensor([[0], [0]])
                het_data['ring', 'r2r', 'ring'].edge_attr = torch.LongTensor([[self_loop_id]]).reshape(-1) # 21 is the index of the self-loop edge type            
            else:
                het_data['ring', 'r2r', 'ring'].edge_index = ring_graphs[idx].edge_index.long()  
                het_data['ring', 'r2r', 'ring'].edge_attr = ring_graphs[idx].edge_attr.long().reshape(-1)       
            # Ring-atom graph
            r2a_edge_index = []
            a2r_edge_index = []
            for ring_id in range(het_data['ring'].num_nodes):
                target_atoms = ring_graphs[idx].ring2atom[ring_graphs[idx].ring2atom_batch==ring_id].tolist()
                r2a_edge_index.append(torch.LongTensor([[ring_id for _ in range(len(target_atoms))], 
                                                                               [atom_id for atom_id in target_atoms]]))
                a2r_edge_index.append(torch.LongTensor([ [atom_id for atom_id in target_atoms],
                                                                               [ring_id for _ in range(len(target_atoms))]]))     
            het_data['ring', 'r2a', 'atom'].edge_index = torch.cat(r2a_edge_index, dim=1)
            het_data['atom', 'a2r', 'ring'].edge_index = torch.cat(a2r_edge_index, dim=1)
            het_data['ring', 'r2a', 'atom'].edge_attr = torch.ones(het_data['ring', 'r2a', 'atom'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data['atom', 'a2r', 'ring'].edge_attr = torch.ones(het_data['atom', 'a2r', 'ring'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data.smiles = data.smiles
            
            
            het_data.ring_atoms = ring_graphs[idx].ring2atom
            het_data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            het_data.ring_atoms_map = ring_graphs[idx].ring2atom_batch            
            
            data_list.append(het_data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
class nNFAHetDataset(InMemoryDataset):
    def __init__(self, root='./data/Polymer_NFA_n/Het/', transform=None, pre_transform=None, pre_filter=None, version='V1'):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'raw/Polymer_NFA_n.csv'

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.version}.pt'
    def download(self):
        pass

    def process(self):
        if self.version == 'V1':
            self_loop_id = 21
        elif self.version == 'V2':
            self_loop_id = 40
        elif self.version == 'V3':
            self_loop_id = 17
        elif self.version == 'V4':
            self_loop_id = 4        
        elif self.version == 'V5':
            self_loop_id = 6
        elif self.version == 'V6':
            self_loop_id = 11      
        elif self.version == 'V7':
            self_loop_id = 10    
        elif self.version == 'V8':
            self_loop_id = 11                                  
        else:
            raise ValueError('Invalid version')          
        # Read data into huge `Data` list.
        data_list = []
        dataset = nNFADataset()
        ring_graphs = torch.load(f'./data/Polymer_NFA_n/ring_graphs_V1.pt')
        for idx, data in enumerate(dataset):
            het_data = HeteroData()
            
            het_data.y = data.y
            # Atom graph
            het_data['atom'].x = data.x
            het_data['atom', 'a2a', 'atom'].edge_index = data.edge_index
            het_data['atom', 'a2a', 'atom'].edge_attr = data.edge_attr
            # Ring graph
            het_data['ring'].x = ring_graphs[idx].x
            if ring_graphs[idx].edge_index.shape[1] == 0:
                het_data['ring', 'r2r', 'ring'].edge_index = torch.LongTensor([[0], [0]])
                het_data['ring', 'r2r', 'ring'].edge_attr = torch.LongTensor([[self_loop_id]]).reshape(-1) # 21 is the index of the self-loop edge type            
            else:
                het_data['ring', 'r2r', 'ring'].edge_index = ring_graphs[idx].edge_index.long()  
                het_data['ring', 'r2r', 'ring'].edge_attr = ring_graphs[idx].edge_attr.long().reshape(-1)       
            # Ring-atom graph
            r2a_edge_index = []
            a2r_edge_index = []
            for ring_id in range(het_data['ring'].num_nodes):
                target_atoms = ring_graphs[idx].ring2atom[ring_graphs[idx].ring2atom_batch==ring_id].tolist()
                r2a_edge_index.append(torch.LongTensor([[ring_id for _ in range(len(target_atoms))], 
                                                                               [atom_id for atom_id in target_atoms]]))
                a2r_edge_index.append(torch.LongTensor([ [atom_id for atom_id in target_atoms],
                                                                               [ring_id for _ in range(len(target_atoms))]]))     
            het_data['ring', 'r2a', 'atom'].edge_index = torch.cat(r2a_edge_index, dim=1)
            het_data['atom', 'a2r', 'ring'].edge_index = torch.cat(a2r_edge_index, dim=1)
            het_data['ring', 'r2a', 'atom'].edge_attr = torch.ones(het_data['ring', 'r2a', 'atom'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data['atom', 'a2r', 'ring'].edge_attr = torch.ones(het_data['atom', 'a2r', 'ring'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            
            het_data.smiles = data.smiles
            
            
            het_data.ring_atoms = ring_graphs[idx].ring2atom
            het_data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            het_data.ring_atoms_map = ring_graphs[idx].ring2atom_batch                
            data_list.append(het_data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
class pNFAHetDataset(InMemoryDataset):
    def __init__(self, root='./data/Polymer_NFA_p/Het/', transform=None, pre_transform=None, pre_filter=None, version='V1'):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'raw/Polymer_NFA_p.csv'

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.version}.pt'
    def download(self):
        pass

    def process(self):
        if self.version == 'V1':
            self_loop_id = 21
        elif self.version == 'V2':
            self_loop_id = 40
        elif self.version == 'V3':
            self_loop_id = 17
        elif self.version == 'V4':
            self_loop_id = 4        
        elif self.version == 'V5':
            self_loop_id = 6
        elif self.version == 'V6':
            self_loop_id = 11      
        elif self.version == 'V7':
            self_loop_id = 10    
        elif self.version == 'V8':
            self_loop_id = 11                                  
        else:
            raise ValueError('Invalid version')     
        # Read data into huge `Data` list.
        data_list = []
        dataset = pNFADataset()
        ring_graphs = torch.load(f'./data/Polymer_NFA_p/ring_graphs_V1.pt')
        for idx, data in enumerate(dataset):
            het_data = HeteroData()
            
            het_data.y = data.y
            # Atom graph
            het_data['atom'].x = data.x
            het_data['atom', 'a2a', 'atom'].edge_index = data.edge_index
            het_data['atom', 'a2a', 'atom'].edge_attr = data.edge_attr
            # Ring graph
            het_data['ring'].x = ring_graphs[idx].x
            if ring_graphs[idx].edge_index.shape[1] == 0:
                het_data['ring', 'r2r', 'ring'].edge_index = torch.LongTensor([[0], [0]])
                het_data['ring', 'r2r', 'ring'].edge_attr = torch.LongTensor([[self_loop_id]]).reshape(-1)    # 21 is the index of the self-loop edge type            
            else:
                het_data['ring', 'r2r', 'ring'].edge_index = ring_graphs[idx].edge_index.long()  
                het_data['ring', 'r2r', 'ring'].edge_attr = ring_graphs[idx].edge_attr.long().reshape(-1)   
            # Ring-atom graph
            r2a_edge_index = []
            a2r_edge_index = []
            for ring_id in range(het_data['ring'].num_nodes):
                target_atoms = ring_graphs[idx].ring2atom[ring_graphs[idx].ring2atom_batch==ring_id].tolist()
                r2a_edge_index.append(torch.LongTensor([[ring_id for _ in range(len(target_atoms))], 
                                                                               [atom_id for atom_id in target_atoms]]))
                a2r_edge_index.append(torch.LongTensor([ [atom_id for atom_id in target_atoms],
                                                                               [ring_id for _ in range(len(target_atoms))]]))     
            het_data['ring', 'r2a', 'atom'].edge_index = torch.cat(r2a_edge_index, dim=1)
            het_data['atom', 'a2r', 'ring'].edge_index = torch.cat(a2r_edge_index, dim=1)
            het_data['ring', 'r2a', 'atom'].edge_attr = torch.ones(het_data['ring', 'r2a', 'atom'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data['atom', 'a2r', 'ring'].edge_attr = torch.ones(het_data['atom', 'a2r', 'ring'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            
            het_data.smiles = data.smiles
            
            
            het_data.ring_atoms = ring_graphs[idx].ring2atom
            het_data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            het_data.ring_atoms_map = ring_graphs[idx].ring2atom_batch                
            data_list.append(het_data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])




class CEPDBHetDataset(InMemoryDataset):
    def __init__(self, root='/data/CEPDB/Het', transform=None, pre_transform=None, pre_filter=None, version='V1'):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'CEPDB.csv'

    @property
    def processed_file_names(self) -> str:
        return f'data.pt'
    def download(self):
        pass

    def process(self):
        if self.version == 'V1':
            self_loop_id = 21
        elif self.version == 'V2':
            self_loop_id = 40
        elif self.version == 'V3':
            self_loop_id = 17
        elif self.version == 'V4':
            self_loop_id = 4        
        elif self.version == 'V5':
            self_loop_id = 6
        elif self.version == 'V6':
            self_loop_id = 11      
        elif self.version == 'V7':
            self_loop_id = 10    
        elif self.version == 'V8':
            self_loop_id = 11                                  
        else:
            raise ValueError('Invalid version')             
        # Read data into huge `Data` list.
        data_list = []
        dataset = CEPDBDataset()
        ring_graphs = CEPDBRingDataset()
        for idx, (data, ring_graph) in enumerate(zip(dataset, ring_graphs)):
            het_data = HeteroData()
            het_data.id = idx
            het_data.y = data.y
            # Atom graph
            het_data['atom'].x = data.x
            het_data['atom', 'a2a', 'atom'].edge_index = data.edge_index
            het_data['atom', 'a2a', 'atom'].edge_attr = data.edge_attr
            # Ring graph
            het_data['ring'].x = ring_graph.ring_x
            if ring_graph.ring_edge_index.shape[1] == 0:
                het_data['ring', 'r2r', 'ring'].edge_index = torch.LongTensor([[0], [0]])
                het_data['ring', 'r2r', 'ring'].edge_attr = torch.LongTensor([[self_loop_id]]).reshape(-1)    # 21 is the index of the self-loop edge type            
            else:
                het_data['ring', 'r2r', 'ring'].edge_index =  ring_graph.ring_edge_index.long()  
                het_data['ring', 'r2r', 'ring'].edge_attr = ring_graph.ring_edge_attr.long().reshape(-1)   
            # Ring-atom graph
            r2a_edge_index = []
            a2r_edge_index = []
            for ring_id in range(het_data['ring'].num_nodes):
                target_atoms = ring_graph.ring_atoms[ring_graph.ring_atoms_map==ring_id].tolist()
                r2a_edge_index.append(torch.LongTensor([[ring_id for _ in range(len(target_atoms))], 
                                                                                [atom_id for atom_id in target_atoms]]))
                a2r_edge_index.append(torch.LongTensor([ [atom_id for atom_id in target_atoms],
                                                                                [ring_id for _ in range(len(target_atoms))]]))     
            het_data['ring', 'r2a', 'atom'].edge_index = torch.cat(r2a_edge_index, dim=1)
            het_data['atom', 'a2r', 'ring'].edge_index = torch.cat(a2r_edge_index, dim=1)
            het_data['ring', 'r2a', 'atom'].edge_attr = torch.ones(het_data['ring', 'r2a', 'atom'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data['atom', 'a2r', 'ring'].edge_attr = torch.ones(het_data['atom', 'a2r', 'ring'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            
            het_data.smiles = data.smiles
            
            
            het_data.ring_atoms = ring_graph.ring_atoms
            het_data.num_ringatoms = ring_graph.ring_atoms.shape[0]   
            het_data.ring_atoms_map = ring_graph.ring_atoms_map              
            data_list.append(het_data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])





def get_dataset_het(args, transform=None):
    meta = {}
    transformer = None
    # Load data
    if args.featurizer == 'MACCS':
        featurizer = dc.feat.MACCSKeysFingerprint()
    elif args.featurizer == 'ECFP6':
        featurizer = dc.feat.CircularFingerprint(size=1024, radius=6)
    elif args.featurizer == 'Mordred':
        featurizer = dc.feat.MordredDescriptors(ignore_3D=True)
    elif args.featurizer is None:
        featurizer = None
    else:
        raise NotImplementedError   
    target_task = args.target_task
    
    if args.dataset == 'HOPV':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = HOPVHetDataset(transform=transform)
        # else:
        dataset_pyg = HOPVHetDataset(transform=transform, version=args.dataset_version)
        index_dir = './data/HOPV/'
        
    elif args.dataset == 'PFD':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = PolymerFAHetDataset(transform=transform)
        # else:
        dataset_pyg = PolymerFAHetDataset(transform=transform, version=args.dataset_version)
        index_dir = './data/Polymer_FA/'

    elif args.dataset == 'PD':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = pNFAHetDataset(transform=transform)
        # else:
        dataset_pyg = pNFAHetDataset(transform=transform, version=args.dataset_version)
        index_dir = './data/Polymer_NFA_p/'
      
    elif args.dataset == 'NFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = nNFAHetDataset(transform=transform)      
        # else:
        dataset_pyg = nNFAHetDataset(transform=transform, version=args.dataset_version) 
        index_dir = './data/Polymer_NFA_n/'
    else:
        raise NotImplementedError  
    
    X = featurizer.featurize(dataset_pyg.data.smiles) if featurizer is not None else np.arange(len(dataset_pyg)).reshape(-1,1)
    meta['fingerprint_dim'] = X.shape[1]
    if args.target_mode == 'single':
        if args.dataset == 'HOPV' and args.target_task == 0:
            nonzero_mask = dataset_pyg.data.y[:, target_task]>-100
        else:
            nonzero_mask = dataset_pyg.data.y[:, target_task]!=0
        smiles = np.array(dataset_pyg.data.smiles)[nonzero_mask].tolist()
        dataset = dc.data.DiskDataset.from_numpy(X[nonzero_mask], dataset_pyg.data.y.numpy()[nonzero_mask, target_task], None, smiles)
        meta['num_classes'] = 1
    elif args.target_mode == 'multi':
        dataset = dc.data.DiskDataset.from_numpy(X, dataset_pyg.data.y, None, dataset_pyg.data.smiles)
        meta['num_classes'] = dataset.y.shape[1]  
    else:
        raise NotImplementedError               
  
    meta['target_task'] = target_task
    # Split dataset
    if args.splitter == 'random':
        splitter = dc.splits.RandomSplitter()
    elif args.splitter == 'scaffold':
        splitter = dc.splits.ScaffoldSplitter()
    else:
        raise NotImplementedError
    # Train: 60%, Valid: 20%, Test: 20%
    train_index, valid_index, test_index = splitter.split(dataset, frac_train=args.frac_train, frac_valid=(1-args.frac_train)/2, frac_test=(1-args.frac_train)/2) 
    train_dataset, valid_dataset, test_dataset = dataset.select(train_index),  dataset.select(valid_index), dataset.select(test_index)

    if args.normalize:
        if args.scaler == 'standard':
            transformer = StandardScaler() 
        elif args.scaler == 'minmax':
            transformer = MinMaxScaler()
        transformer.fit(train_dataset.y.reshape(-1,meta['num_classes']))
        y_train = transformer.transform(train_dataset.y.reshape(-1,meta['num_classes']))
        y_valid = transformer.transform(valid_dataset.y.reshape(-1,meta['num_classes']))
        y_test = transformer.transform(test_dataset.y.reshape(-1,meta['num_classes']))
    else:
        y_train = train_dataset.y.reshape(-1,meta['num_classes'])
        y_valid = valid_dataset.y.reshape(-1,meta['num_classes'])
        y_test = test_dataset.y.reshape(-1,meta['num_classes'])
        
        
    data_list_train = []
    for idx, fingerprint, y in zip(train_index, train_dataset.X, y_train):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.y = torch.FloatTensor(y).reshape(1,-1)

        
        data_list_train.append(data)
    
    data_list_valid = []
    for idx, fingerprint, y in zip(valid_index, valid_dataset.X, y_valid):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.y = torch.FloatTensor(y).reshape(1,-1)

                
        data_list_valid.append(data)
        
    
    data_list_test = []
    for idx, fingerprint, y in zip(test_index, test_dataset.X, y_test):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.y = torch.FloatTensor(y).reshape(1,-1)

                
        data_list_test.append(data)
    

    dataloader = DataLoader(data_list_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(data_list_valid, batch_size=1024, shuffle=False) 
    dataloader_test = DataLoader(data_list_test, batch_size=1024, shuffle=False)
    
    
    
    return dataloader,  dataloader_test, dataloader_val, transformer, meta
