import numpy as np
from collections import defaultdict

import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import Crippen, rdMolDescriptors
# -----------------------------
# Atom featurizer
# -----------------------------
ATOM_LIST = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H, B, C, N, O, F, Si, P, S, Cl, Br, I
HYBRIDIZATION_LIST = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]


_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
]

_BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS
]


def one_hot(x, choices):
    vec = [0]*len(choices)
    try:
        idx = choices.index(x)
        vec[idx] = 1
    except ValueError:
        pass
    return vec


def is_benzylic(atom: rdchem.Atom, mol: Chem.Mol) -> float:
    if atom.GetHybridization() != rdchem.HybridizationType.SP3:
        return 0.0
    for b in atom.GetBonds():
        j = b.GetOtherAtom(atom).GetIdx()
        if mol.GetAtomWithIdx(j).GetIsAromatic():
            return 1.0
    return 0.0


def hetero_neighbor_count(atom: rdchem.Atom) -> float:
    hetero = {7, 8, 16, 17, 35, 53}
    cnt = sum(1 for nbr in atom.GetNeighbors()
              if nbr.GetAtomicNum() in hetero)
    return min(cnt, 4) / 4.0  


def is_basic_n(atom: rdchem.Atom) -> float:

    if atom.GetAtomicNum() != 7:
        return 0.0

    if atom.GetFormalCharge() > 0:
        return 1.0
    if atom.GetTotalDegree() >= 3 and atom.GetHybridization() in (
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP2,
    ):
        return 1.0
    return 0.0


def branching_degree(atom: rdchem.Atom) -> float:
    deg = 0
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() != 1:
            deg += 1
    return min(deg, 4) / 4.0 


def atom_features(
    atom: rdchem.Atom,
    mol: Chem.Mol = None,
    logp_contrib: float = 0.0,
    tpsa_contrib: float = 0.0,
):
    Z = atom.GetAtomicNum()
    degree = min(atom.GetTotalDegree(), 5)
    formal_charge = atom.GetFormalCharge()
    hyb = atom.GetHybridization()
    aromatic = atom.GetIsAromatic()
    in_ring = atom.IsInRing()
    mass = atom.GetMass()*0.01 

    feats = []
    feats += one_hot(Z, ATOM_LIST)                    
    feats += one_hot(degree, list(range(6)))           
    feats += [formal_charge/3.0]                        
    feats += one_hot(hyb, HYBRIDIZATION_LIST)          
    feats += [1.0 if aromatic else 0.0]                 
    feats += [1.0 if in_ring else 0.0]                  
    feats += [mass]                                     

    if mol is not None:
        feats += [is_benzylic(atom, mol)]               
    else:
        feats += [0.0]

    feats += [hetero_neighbor_count(atom)]              
    feats += [is_basic_n(atom)]                        
    feats += [branching_degree(atom)]                   

    feats += [logp_contrib / 2.0]                       
    feats += [tpsa_contrib / 50.0]                      

    return feats



def has_som_label_graph(g):
    return int((g.y > 0.5).sum().item() > 0)


def make_strata_labels(graphs):
    pairs = []
    for g in graphs:
        c = int(g.cyp_idx[0].item())
        has = int((g.y > 0.5).sum().item() > 0)
        pairs.append((c, has))
    uniq = {p: i for i, p in enumerate(sorted(set(pairs)))}
    return np.array([uniq[p] for p in pairs], dtype=int)

def bond_one_hot(idx, size):
    v = [0.0] * size
    if 0 <= idx < size:
        v[idx] = 1.0
    return v

def bond_features(bond: Chem.rdchem.Bond):
    if bond is None:
        return [0.0] * (len(_BOND_TYPES) + 2 + len(_BOND_STEREOS))
    f = []
    btype = bond.GetBondType()
    bt_idx = _BOND_TYPES.index(btype) if btype in _BOND_TYPES else len(_BOND_TYPES)
    f += bond_one_hot(bt_idx, len(_BOND_TYPES) + 1)
    f += [1.0 if bond.GetIsConjugated() else 0.0]
    f += [1.0 if bond.IsInRing() else 0.0]
    st = bond.GetStereo()
    st_idx = _BOND_STEREOS.index(st) if st in _BOND_STEREOS else 0
    f += bond_one_hot(st_idx, len(_BOND_STEREOS))
    return f


def mol_to_graph(mol, smiles):
    num_atoms = mol.GetNumAtoms()

    som_idxs = []
    if mol.HasProp('PRIMARY_SOM'):
        primary_som = mol.GetProp('PRIMARY_SOM').strip()
        som_idxs = [int(x)-1 for x in primary_som.split() if x.isdigit()]
    labels = torch.zeros(num_atoms, dtype=torch.float32)
    for idx in som_idxs:
        if 0 <= idx < num_atoms:
            labels[idx] = 1.0

    crippen_contribs = Crippen._GetAtomContribs(mol)
    tpsa_contribs = rdMolDescriptors._CalcTPSAContribs(mol)

    x_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        logp_i = crippen_contribs[i][0] if i < len(crippen_contribs) else 0.0
        tpsa_i = tpsa_contribs[i] if i < len(tpsa_contribs) else 0.0
        feats = atom_features(atom, mol=mol,
                              logp_contrib=logp_i,
                              tpsa_contrib=tpsa_i)
        x_list.append(feats)
    x = torch.tensor(x_list, dtype=torch.float32)

    edges, eattr = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edges.append((i, j))
        edges.append((j, i))
        eattr += [bf, bf]
    if len(edges) == 0:
        edge_index = torch.empty(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(eattr, dtype=torch.float32) if len(eattr) > 0 else \
        torch.zeros((0, len(_BOND_TYPES) + 1 + 2 + len(_BOND_STEREOS)), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=labels, smiles=smiles, num_atoms=num_atoms)
    return data, som_idxs



def split_train_val_by_index_stratified_cyp(graphs, indices, val_ratio=0.1, seed=1):
    rng = np.random.RandomState(seed)
    by_key = defaultdict(list)
    for idx in indices:
        c = int(graphs[idx].cyp_idx[0].item())
        has = int((graphs[idx].y > 0.5).sum().item() > 0)
        by_key[(c, has)].append(idx)
    train_idx, val_idx = [], []
    for _, idxs in by_key.items():
        idxs = np.array(idxs); rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs)*val_ratio))) if len(idxs) > 1 else (1 if len(val_idx)==0 else 0)
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())
    return train_idx, val_idx

