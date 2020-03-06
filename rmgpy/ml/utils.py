import contextlib
import os
from argparse import Namespace
from typing import Callable, Union
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, rdchem
from rdkit import RDConfig
import numpy as np 
import os.path as osp
try:
    import dgl
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    from dgl.data.chem.utils import mol_to_complete_graph, mol_to_bigraph
    from dgl import BatchedDGLGraph
    from dgl import backend as dgl_F
    from dgl.nn.pytorch import Set2Set, NNConv, SetTransformerDecoder, AvgPooling, SumPooling, MaxPooling, SortPooling
except ImportError as e:
    dgl = None
    dgl_exception = e

def atom_features(mol):
    """Featurization for all atoms in a molecule. The atom indices
    will be preserved.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object

    Returns
    -------
    atom_feats_dict : dict
        Dictionary for atom features
    """
    atom_feats_dict = defaultdict(list)
    is_donor = defaultdict(int)
    is_acceptor = defaultdict(int)

    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    mol_feats = mol_featurizer.GetFeaturesForMol(mol)
    

    for i in range(len(mol_feats)):
        if mol_feats[i].GetFamily() == 'Donor':
            node_list = mol_feats[i].GetAtomIds()
            for u in node_list:
                is_donor[u] = 1
        elif mol_feats[i].GetFamily() == 'Acceptor':
            node_list = mol_feats[i].GetAtomIds()
            for u in node_list:
                is_acceptor[u] = 1

    num_atoms = mol.GetNumAtoms()
    for u in range(num_atoms):
        atom = mol.GetAtomWithIdx(u)
        symbol = atom.GetSymbol()
        atom_type = atom.GetAtomicNum()
        aromatic = atom.GetIsAromatic()
        hybridization = atom.GetHybridization()
        num_h = atom.GetTotalNumHs()
        is_in_ring = 1 if rdchem.Atom.IsInRing(atom) else 0 
        atom_feats_dict['node_type'].append(atom_type)

        h_u = []
        h_u += [int(symbol == x) for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']]
        h_u.append(atom_type)
        h_u.append(is_acceptor[u])
        h_u.append(is_donor[u])
        h_u.append(int(aromatic))
        h_u += [
            int(hybridization == x)
            for x in (Chem.rdchem.HybridizationType.SP,
                      Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3)
        ]
        h_u.append(num_h)
        h_u.append(is_in_ring)
        atom_feats_dict['n_feat'].append(
            dgl_F.tensor(np.array(h_u).astype(np.float32)))

    atom_feats_dict['n_feat'] = dgl_F.stack(atom_feats_dict['n_feat'], dim=0)
    atom_feats_dict['node_type'] = dgl_F.tensor(np.array(
        atom_feats_dict['node_type']).astype(np.int64))

    return atom_feats_dict


def edge_features(mol, self_loop=False):
    """Featurization for all bonds in a molecule.
    The bond indices will be preserved.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object
    self_loop : bool
        Whether to add self loops. Default to be False.

    Returns
    -------
    bond_feats_dict : dict
        Dictionary for bond features
    """
    bond_feats_dict = defaultdict(list)

    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        bond_type = bond.GetBondType()
        bond_feats_dict['e_feat'].append(np.array([
                float(bond_type == x)
                for x in (Chem.rdchem.BondType.SINGLE,
                            Chem.rdchem.BondType.DOUBLE,
                            Chem.rdchem.BondType.TRIPLE,
                            Chem.rdchem.BondType.AROMATIC)
            ]))
        bond_feats_dict['e_feat'].append(np.zeros(4)) #adding zeros for v,u
    
    bond_feats_dict['e_feat'] = dgl_F.tensor(
        np.array(bond_feats_dict['e_feat']).astype(np.float32))
    

    return bond_feats_dict



def make_dgl_graph_from_smiles(smiles: str) -> Callable[[str], dgl.BatchedDGLGraph]:
    mol = Chem.MolFromSmiles(smiles)
    dgl_graph = mol_to_bigraph(mol,atom_featurizer=atom_features,
                                bond_featurizer=edge_features)
    bg = dgl.batch([dgl_graph])
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return bg


class MPNPool(torch.nn.Module):
    def __init__(self,
                 node_input_dim=16,
                 edge_input_dim=4,
                 output_dim=1,
                 node_hidden_dim=64,
                 edge_hidden_dim=128,
                 num_step_message_passing=5,
                 pooling='Avg',
                 num_step_set2set=12,
                 num_layer_set2set=1,
                 num_heads=6,
                 d_head=10,
                 d_ff=6,
                 n_layers=5,
                 k=4,
                 dropouth=0.0,
                 dropouta=0.0
                 ):
        super(MPNPool, self).__init__()
        self.node_input_dim           =node_input_dim          
        self.edge_input_dim           =edge_input_dim          
        self.output_dim               =output_dim              
        self.node_hidden_dim          =node_hidden_dim         
        self.edge_hidden_dim          =edge_hidden_dim         
        self.num_step_message_passing =num_step_message_passing
        self.pooling                  =pooling                 
        self.num_step_set2set         =num_step_set2set        
        self.num_layer_set2set        =num_layer_set2set       
        self.num_heads                =num_heads               
        self.d_head                   =d_head                  
        self.d_ff                     =d_ff                    
        self.n_layers                 =n_layers                
        self.k                        =k                       
        self.dropouth                 =dropouth                
        self.dropouta                 =dropouta 
        self.message_passing_module   =MPN(node_input_dim=self.node_input_dim, 
                                          edge_input_dim=self.edge_input_dim,      
                                          node_hidden_dim=self.node_hidden_dim,
                                          edge_hidden_dim=self.edge_hidden_dim,
                                          num_step_message_passing=self.num_step_message_passing)
        if self.pooling=='set2set':
            self.pooling_module = Set2Set(self.node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)
            self.lin1 = torch.nn.Linear(2* self.node_hidden_dim, 200)
            
        elif self.pooling=='set_attention':
            self.pooling_module = SetTransformerDecoder(self.node_hidden_dim,self.num_heads,self.d_head,self.d_ff,
                                                        self.n_layers,self.k,self.dropouth,self.dropouta)
            self.lin1 = torch.nn.Linear(self.k * self.node_hidden_dim, 200)
            
        elif self.pooling =='avg':
            self.pooling_module = AvgPooling()
            self.lin1 = torch.nn.Linear(self.node_hidden_dim, 200)
           
        elif self.pooling=='sum':
            self.pooling_module = SumPooling()
            self.lin1 = torch.nn.Linear(self.node_hidden_dim, 200)
            
        elif self.pooling=='max':
            self.pooling_module = MaxPooling()
            self.lin1 = torch.nn.Linear(self.node_hidden_dim, 200)    
        elif self.pooling=='sort':
            self.pooling_module = SortPooling()
            self.lin1 = torch.nn.Linear(self.node_hidden_dim, 200)
            
        self.lin2 = torch.nn.Linear(200, self.output_dim)
        
        
    def forward(self,g):
        node_rep = self.message_passing_module(g)
        graph_rep = self.pooling_module(g,node_rep)
        preds = F.relu(self.lin1(graph_rep))
        
        preds = self.lin2(preds)
        
        return preds