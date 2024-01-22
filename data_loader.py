import os
import re
import os.path as osp
from scipy import sparse as sp
import torch
import numpy as np
import networkx as nx
from torch_geometric.loader import DataLoader, DenseDataLoader
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
import torch_geometric.transforms as T
from torch_geometric.utils import degree, from_smiles

import sys

import deepchem as dc


import torch
from torch_geometric.data import InMemoryDataset, download_url
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from transform import *

class HOPVDataset(InMemoryDataset):
    """HOMO [a.u.], LUMO [a.u.], Electrochemical gap [a.u.] = band gap = LUMO-HOMO gap, 
    Optical gap [a.u.] (Eg)= LUMO-HOMO gap an environment with photom, Power conversion efficiency [%], 
    Open circuit potential [V] (Voc), Short circuit current density [mA/cm^2] (Jsc), and fill factor [%] (FF)
    
    final y:
    PCE, HOMO, LUMO, band gap, Voc, Jsc, FF
    """
    def __init__(self, root='./data/HOPV/', transform=None, pre_transform=None, pre_filter=None, version=None):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        return ''
    @property
    def processed_file_names(self) -> str:
        if self.version is None:
            return 'data.pt'
        else:
            return f'data_{self.version}.pt'
    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        featurizer = dc.feat.MACCSKeysFingerprint()
        tasks, datasets, transformers = dc.molnet.load_hopv(featurizer=featurizer, splitter=None, transformers=[])
        dataset = datasets[0]
        data_list = []
        for smiles, y in zip(dataset.ids, dataset.y):
            data = from_smiles(smiles)
            data['y'] = torch.as_tensor([y[4], np.abs(y[0]), np.abs(y[1]), np.abs(y[2]), y[5], y[6], float(y[7])/100]).view(1, -1).to(torch.float32)
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        
        
class PolymerFADataset(InMemoryDataset):
    # PCE, HOMO, LUMO, band gap, Voc, Jsc, FF
    def __init__(self, root='./data/PFD/', transform=None, pre_transform=None, pre_filter=None, version=None):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'PFD.csv'

    @property
    def processed_file_names(self) -> str:
        if self.version is None:
            return 'data.pt'
        else:
            return f'data_{self.version}.pt'
    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv( os.path.join(self.root, self.raw_file_names))
        data_list = []
        for nickname, pce, pce_avg, Voc, Jsc, FF, Mw, Mn,PDI, Monomer, bandgap, smiles, HOMO, LUMO in df.values:
            pce= float(pce)
            data = from_smiles(smiles)
            data['name'] = nickname
            data['y'] = torch.as_tensor([pce, np.abs(HOMO), np.abs(LUMO), np.abs(bandgap), Voc, Jsc, FF]).view(1, -1).to(torch.float32)
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        
class nNFADataset(InMemoryDataset):
    # PCE, HOMO, LUMO, band gap = Eg, Voc, Jsc, FF
    def __init__(self, root='./data/NFA/', transform=None, pre_transform=None, pre_filter=None, version=None):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'NFA.csv'

    @property
    def processed_file_names(self) -> str:
        if self.version is None:
            return 'data.pt'
        else:
            return f'data_{self.version}.pt'
        
    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv( os.path.join(self.root, self.raw_file_names))
        data_list = []
        for smiles, pce, pce_avg, Jsc, FF, Voc, Eg_n, M, HOMO, LUMO in df.values:
            pce= float(pce)
            data = from_smiles(smiles)
            data['y'] = torch.as_tensor([pce, np.abs(HOMO), np.abs(LUMO), np.abs(Eg_n), Voc, Jsc, FF]).view(1, -1).to(torch.float32)
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        
class pNFADataset(InMemoryDataset):
    def __init__(self, root='./data/PD/', transform=None, pre_transform=None, pre_filter=None, version=None):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'PD.csv'

    @property
    def processed_file_names(self) -> str:
        if self.version is None:
            return 'data.pt'
        else:
            return f'data_{self.version}.pt'
        
    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv( os.path.join(self.root, self.raw_file_names))
        data_list = []
        for smiles, pce, pce_avg, Jsc, FF, Voc, Eg_n, Mw, Mn, PDI, HOMO, LUMO in df.values:
            pce= float(pce)
            data = from_smiles(smiles)
            data['y'] = torch.as_tensor([pce, np.abs(HOMO), np.abs(LUMO), np.abs(Eg_n), Voc, Jsc, FF]).view(1, -1).to(torch.float32)
            data_list.append(data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        
        
class CEPDBDataset(InMemoryDataset):
    def __init__(self, root='/data/CEPDB/', transform=None, pre_transform=None, pre_filter=None, version=None):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'CEPDB.csv'

    @property
    def processed_file_names(self) -> str:
        if self.version is None:
            return 'data.pt'
        else:
            return f'data_{self.version}.pt'
    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv( os.path.join(self.root, self.raw_file_names))
        data_list = []
        for id, (molgraph_id, smiles, stoich, n_el, n_bf_sz, n_bf_dzp,n_bf_tzp, mass, pce, voc, jsc, homo_average, lumo_average,homo_max, lumo_max, homo_min, lumo_min) in enumerate(df.values):
            data = from_smiles(smiles)
            data['molgraph_id'] = molgraph_id
            data['id'] = id
            data.y = torch.as_tensor([np.abs(pce), np.abs(homo_average), np.abs(lumo_average), np.abs(homo_average-lumo_average), np.abs(voc), np.abs(jsc)]).view(1, -1).to(torch.float32)
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
