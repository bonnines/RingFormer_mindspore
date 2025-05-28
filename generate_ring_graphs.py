from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.utils import degree, from_smiles, to_networkx
import shutil, os
import os.path as osp
import torch
import numpy as np
from tqdm import tqdm
# from OGNN.dualgraph.dataset import DGData
# from OGNN.dualgraph.mol import smiles2graphwithface
from rdkit import Chem
from copy import deepcopy
import os
import numpy as np
import pandas as pd
from typing import List
from deepchem.molnet import *
from rdkit import Chem
import csv
import pandas as pd
from data_loader import *
from rdkit.Chem import BRICS
from rdkit.Chem import GetSymmSSSR
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole, MolsToGridImage
from rdkit.Chem import Draw

import json
from collections import defaultdict
import argparse


def subMol(mol, match):
	#not sure why this functionality isn't implemented natively
	#but get the interconnected bonds for the match
	atoms = set(match)
	bonds = set()
	for a in atoms:
		atom = mol.GetAtomWithIdx(a)
		for b in atom.GetBonds():
			if b.GetOtherAtomIdx(a) in atoms:
				bonds.add(b.GetIdx())
	return Chem.PathToSubmol(mol,list(bonds))


def sub_to_smiles(mol, atoms):
	frag = subMol(mol, atoms)
	try:
		Chem.SanitizeMol(frag)
	except:
		frag = frag
	if frag is None:
		frag = subMol(mol, atoms)
	smiles = Chem.MolToSmiles(frag)
	return smiles

def is_graph_connected(data):
    # Convert edge_index to networkx format
    G = nx.Graph()
    for i in range(data.edge_index.shape[1]):
        src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        G.add_edge(src, dst)
    for i in range(data.x.shape[0]):
        G.add_node(i)

    # Check connectivity
    return nx.is_connected(G)

def shortest_path_between_rings(mol_G, degree_vector, ring1, ring2):
    
    # Find the atoms that form the rings
    ring1_atoms = ring1[degree_vector[ring1]>2].tolist() # only those atoms with degree > 2 are potential
    ring2_atoms = ring2[degree_vector[ring2]>2].tolist()
    
    # Find the shortest path between the rings
    shortest_path = None
    shortest_distance = float('inf')
    
    for atom1 in ring1_atoms:
        for atom2 in ring2_atoms:
            try:
                path = nx.bidirectional_shortest_path(mol_G, source=atom1, target=atom2)
                if len(path) < shortest_distance:
                    shortest_distance = len(path)
                    shortest_path = path
            except nx.NetworkXNoPath:
                # If there's no path between the current pair of atoms, continue to the next pair
                continue
                
    # Return the shortest path in the desired format
    return shortest_path


if __name__ == "__main__":
    # V1
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--dataset', type=str, default='HOPV')
    args = argparse.parse_args()
    ring_edge_attr_dict = {'one-bond': 0,
    '2-atoms': 1,
    '4-atoms': 2,
    '3-atoms': 3,
    '1-atoms': 4,
    '6-6': 5,
    '7': 6,
    '6': 7,
    '14': 8,
    '8-6-6-6-6-6-6-8': 9,
    '6-6-6-6': 10,
    '6-6-6-6-6-6-6-6-6-6-6-6': 11,
    '7-6': 12,
    '6-7': 13,
    '6-6-6-6-6-6': 14,
    '8': 15,
    '32': 16,
    '15': 17}
    
    
    ring_edge_attr_count_dict = {type: 0 for type in ring_edge_attr_dict.keys()}
    if args.dataset == 'HOPV': # TODO
        dataset = HOPVDataset()
        ring_graph_filename = './data/HOPV/ring_graphs_V1.pt' 
        ring_dict_filename = "./data/HOPV/ringID2smiles_V1.pt"
    smiles2id = {}
    id2smiles = {}
    ring_graphs = []
    bond_type_dict = {}
    ring_graphs = []
    for gid, data in enumerate(dataset):
        mol = Chem.MolFromSmiles(data.smiles)
        ring_info = list(GetSymmSSSR(mol))
        n_rings = len(ring_info)
        ring_x = torch.zeros(n_rings, 1)
        row = []
        col = []
        ring_edge_attr = []

        # Ring information
        rings_smiles = []
        rings_ids = []
        # ring-atom mapping
        atom2ring = defaultdict(list)
        ring2atom = []
        ring2atom_batch = [] # Same as ring2edge_batch since the number of rings atoms is the same as the number of ring edges
        # ring-edge mapping
        ring2edge = []
        # 
        re2atom = []
        re2atom_batch = []
        re2edge = []
        re2edge_batch = []
        
        
        edge2ringedge = torch.ones(data.edge_index.shape[1]).long()*-1
        skip_count = 0
                
    for gid, data in enumerate(dataset):
        mol = Chem.MolFromSmiles(data.smiles)
        ring_info = list(GetSymmSSSR(mol))
        n_rings = len(ring_info)
        ring_x = torch.zeros(n_rings, 1)
        row = []
        col = []
        ring_edge_attr = []

        # Ring information
        rings_smiles = []
        rings_ids = []
        # ring-atom mapping
        atom2ring = defaultdict(list)
        ring2atom = []
        ring2atom_batch = [] # Same as ring2edge_batch since the number of rings atoms is the same as the number of ring edges
        # ring-edge mapping
        ring2edge = []
        # 
        re2atom = []
        re2atom_batch = []
        re2edge = []
        re2edge_batch = []
        
        
        edge2ringedge = torch.ones(data.edge_index.shape[1]).long()*-1
        skip_count = 0
                
        for i in range(n_rings):
            ring_i = list(ring_info[i])
            smiles_i = sub_to_smiles(mol, ring_i)
            if smiles_i not in smiles2id:
                smiles2id[smiles_i] = len(smiles2id)
            id_i = smiles2id[smiles_i]
            id2smiles[id_i] = smiles_i    
            ring_x[i] = id_i  
            ring2atom = ring2atom + ring_i
            ring2atom_batch = ring2atom_batch + [i]*len(ring_i)
            for atom in ring_i:
                atom2ring[atom].append(i)
            
            for atom_id, atom in enumerate(ring_i):
                bond_id = ((data.edge_index[0]==ring_i[atom_id-1]) & (data.edge_index[1]==atom)).nonzero()[0][0].item()
                ring2edge.append(bond_id)
        ring2edge = torch.tensor(ring2edge).long()
        ring2atom = torch.tensor(ring2atom).long()
        ring2atom_batch = torch.tensor(ring2atom_batch).long()  
        
            
        for i in range(n_rings):
            ring_i = list(ring_info[i])
            for j in range(i+1, n_rings):
                ring_j = list(ring_info[j])
                common_atoms = list(set(ring_i) & set(ring_j))
                n_common_atoms = len(common_atoms)
                if n_common_atoms >0: # Type 1: two rings share two atoms
                    row.append(i)
                    col.append(j)
                    row.append(j)
                    col.append(i)         
                    ring_bond_edge_types = []      
                    for atomid_i in range(n_common_atoms-1):
                        atom_src = common_atoms[atomid_i]
                        for atomid_j in range(atomid_i+1, n_common_atoms):
                            atom_dst = common_atoms[atomid_j]
                            target_edge = ((data.edge_index[0]==atom_src) & (data.edge_index[1]==atom_dst)).nonzero()
                            if len(target_edge) == 0:
                                continue
                            else:
                                bond_type = mol.GetBondBetweenAtoms(atom_src,atom_dst).GetBondType()
                                if bond_type not in bond_type_dict:
                                    bond_type_dict[bond_type] = len(bond_type_dict)
                                ring_bond_edge_types.append(bond_type_dict[bond_type])
                                edge_id = target_edge[0][0].item()
                                re2edge.append(edge_id)
                                re2edge.append(edge_id)
                                re2edge_batch.append(len(row)-2)
                                re2edge_batch.append(len(row)-1)   
                                break   
                    ring_bond_atom_types = [] 
                    for atom in common_atoms:
                        ring_bond_atom_types.append(mol.GetAtoms()[atom].GetAtomicNum())
                        re2atom.append(atom)
                        re2atom.append(atom)
                        re2atom_batch.append(len(row)-2)
                        re2atom_batch.append(len(row)-1)
                    ring_bond_atom_types = ",".join([str(x) for x in sorted(ring_bond_atom_types)])
                    ring_bond_edge_types = ",".join([str(x) for x in sorted(ring_bond_edge_types)])
                    ring_bond_type = f'{ring_bond_atom_types}-{ring_bond_edge_types}'
                    if ring_bond_type not in ring_edge_attr_dict:
                        ring_edge_attr_dict[ring_bond_type] = len(ring_edge_attr_dict)
                    ring_edge_attr.append(ring_edge_attr_dict[ring_bond_type])  
                    ring_edge_attr.append(ring_edge_attr_dict[ring_bond_type])  
                    
                    
        visited = set()
        for i,j in data.edge_index.T.tolist():
            if f'{i}-{j}' in visited or f'{j}-{i}' in visited:
                continue
            visited.add(f'{i}-{j}')
            if len(atom2ring[i])>1 or len(atom2ring[j])>1 or len(atom2ring[i])==0 or len(atom2ring[j])==0: 
                continue # if the atom in the bond is not in a ring or is in more than one ring, skip
            ring0 = atom2ring[i][0]
            ring1 = atom2ring[j][0]
            if ring0 != ring1: # Type 0: two rings connected by a non-aromatic bond
                row.append(ring0)
                col.append(ring1)
                row.append(ring1)
                col.append(ring0)            
                
                bond_type = mol.GetBondBetweenAtoms(i,j).GetBondType()
                if bond_type not in bond_type_dict:
                    bond_type_dict[bond_type] = len(bond_type_dict)    
                        
                edge_id = ((data.edge_index[0]==i) & (data.edge_index[1]==j)).nonzero()[0][0].item()
                re2edge.append(edge_id)
                re2edge.append(edge_id)
                re2edge_batch.append(len(row)-2)
                re2edge_batch.append(len(row)-1)            
                re2atom.append(i)
                re2atom.append(j)
                re2atom_batch.append(len(row)-2)
                re2atom_batch.append(len(row)-2)
                re2atom.append(i)
                re2atom.append(j)            
                re2atom_batch.append(len(row)-1)            
                re2atom_batch.append(len(row)-1)   

                ring_bond_type = f'one-bond:{bond_type_dict[bond_type]}'
                if ring_bond_type not in ring_edge_attr_dict:
                    ring_edge_attr_dict[ring_bond_type] = len(ring_edge_attr_dict)                    
                ring_edge_attr.append(ring_edge_attr_dict[ring_bond_type]) 
                ring_edge_attr.append(ring_edge_attr_dict[ring_bond_type])  
        re2atom = torch.tensor(re2atom).long()
        re2atom_batch = torch.tensor(re2atom_batch).long()
        re2edge = torch.tensor(re2edge).long()
        re2edge_batch = torch.tensor(re2edge_batch).long()
        
        ring_g = Data(x=ring_x.reshape(-1,1).long(), edge_index=torch.tensor([row, col]), edge_attr=torch.tensor(ring_edge_attr).reshape(-1,1).long(), 
                    ring2atom=ring2atom, ring2edge=ring2edge, ring2atom_batch=ring2atom_batch, re2atom=re2atom, 
                    re2atom_batch=re2atom_batch, re2edge=re2edge, re2edge_batch=re2edge_batch)
        ring_graphs.append(ring_g)

    torch.save(ring_graphs, ring_graph_filename) 
    with open(ring_dict_filename, "w") as json_file:
        json.dump(id2smiles, json_file)