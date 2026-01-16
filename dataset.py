# # dataset.py

import os
import torch
from rdkit import Chem
from collections import defaultdict
from dataset_utils import mol_to_graph
from tqdm import tqdm

def load_cyp_sdf(path):
    suppl = Chem.SDMolSupplier(path)
    mols_data = []
    for mol in suppl:
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol)
        
        som_idxs = []
        if mol.HasProp('PRIMARY_SOM'):
            primary_som = mol.GetProp('PRIMARY_SOM').strip()
            som_idxs = [int(x) for x in primary_som.split() if x.isdigit()]
        
        mols_data.append({
            'mol': mol,
            'smiles': smi,
            'som_idxs': som_idxs
        })
    
    return mols_data


def load_multi_cyp(base_dir, cyp_list):
    cyp2idx = {c: i for i, c in enumerate(cyp_list)}
    num_cyps = len(cyp_list)
    

    mol_annotations = defaultdict(lambda: defaultdict(list))
    mol_objects = {} 
    
    print("Loading molecules and collecting annotations...")
    for cyp in cyp_list:
        sdf_path = os.path.join(base_dir, f'{cyp}.sdf')
        if not os.path.exists(sdf_path):
            print(f"[WARN] Missing {sdf_path}, skipping {cyp}")
            continue
        
        mols_data = load_cyp_sdf(sdf_path)
        print(f"  {cyp}: {len(mols_data)} molecules")
        
        for mol_data in mols_data:
            smi = mol_data['smiles']
            
            if smi not in mol_objects:
                mol_objects[smi] = mol_data['mol']
            
            mol_annotations[smi][cyp] = mol_data['som_idxs']
    
    print(f"Total unique molecules: {len(mol_objects)}")

    all_graphs = []
    
    for cyp in cyp_list:
        sdf_path = os.path.join(base_dir, f'{cyp}.sdf')
        if not os.path.exists(sdf_path):
            continue
        
        mols_data = load_cyp_sdf(sdf_path)
        
        for mol_data in tqdm(mols_data):
            mol = mol_data['mol']
            smi = mol_data['smiles']
            cyp_idx = cyp2idx[cyp]
            
            data, som_idxs = mol_to_graph(mol, smi)
            num_atoms = data.num_nodes
            
            som_annotations = torch.zeros(num_atoms, num_cyps, dtype=torch.float32)
            som_mask = torch.zeros(num_atoms, num_cyps, dtype=torch.float32)

            for other_cyp_idx, other_cyp in enumerate(cyp_list):
                if other_cyp in mol_annotations[smi]:
                    som_mask[:, other_cyp_idx] = 1.0
                    
                    som_list = mol_annotations[smi][other_cyp]
                    for atom_idx_1based in som_list:
                        atom_idx_0based = atom_idx_1based - 1
                        if 0 <= atom_idx_0based < num_atoms:
                            som_annotations[atom_idx_0based, other_cyp_idx] = 1.0

            data.som_annotations = som_annotations
            data.som_mask = som_mask

            data.cyp_idx = torch.full((num_atoms,), cyp_idx, dtype=torch.long)
            data.cyp_name = cyp
            
            all_graphs.append(data)
    
    all_graphs = [g for g in all_graphs if g.num_nodes > 0]
    
    print(f"\nLoaded total graphs: {len(all_graphs)}")
    print(f"CYPs: {cyp_list}")
    

    if len(all_graphs) > 0:
        sample = all_graphs[0]
        print(f"\nSample graph verification:")
        print(f"  num_nodes: {sample.num_nodes}")
        print(f"  som_annotations shape: {sample.som_annotations.shape}")
        print(f"  Expected shape: ({sample.num_nodes}, {num_cyps})")
        print(f"  cyp_idx shape: {sample.cyp_idx.shape}")
        print(f"  Current CYP: {sample.cyp_name}")
        print(f"  SoM in current CYP: {sample.y.sum().item()}")
        print(f"  SoM annotations sum per CYP: {sample.som_annotations.sum(dim=0).tolist()}")
    
    return all_graphs


def apply_no_leakage_to_dataloaders(train_set, test_set, cyp_list):
    cyp2idx = {c: i for i, c in enumerate(cyp_list)}
    
    test_molecule_cyp_pairs = set()
    for g in test_set:
        if hasattr(g, 'smiles') and hasattr(g, 'cyp_name'):
            test_molecule_cyp_pairs.add((g.smiles, g.cyp_name))
    
    print(f"\n[Data Leakage Fix with Masking]")
    
    masked_count = 0
    for g in train_set:
        num_atoms = g.num_nodes
        num_cyps = len(cyp_list)

        if not hasattr(g, 'som_mask'):
            g.som_mask = torch.ones(num_atoms, num_cyps) 

        for c_idx, c_name in enumerate(cyp_list):
            mol_cyp_pair = (g.smiles, c_name)
            
            if mol_cyp_pair in test_molecule_cyp_pairs:
                g.som_mask[:, c_idx] = 0.0 
                masked_count += 1

    print(f"Masked {masked_count} (Molecule, CYP) pairs in Train set to prevent leakage.")
    return train_set