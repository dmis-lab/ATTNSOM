"""Dataset loading for ATTNSOM.

Two benchmarks are supported:

* **Zaretzki dataset** -- one SDF per CYP isoform (``<isoform>.sdf``) with SoM
  atoms stored in the ``PRIMARY_SOM`` SD property using **1-based** indices.
* **AZ-ExactSOM** -- a single CSV (``az_120_compounds.csv``) with SoM atom
  groups and an exact/extended flag per group, using **0-based** indices.
  Only exact annotations are used, as described in the paper.
"""

import ast
import os

import pandas as pd
import torch
from collections import defaultdict
from rdkit import Chem
from tqdm import tqdm

from dataset_utils import mol_to_graph

AZ_CSV_NAME = 'az_120_compounds.csv'
AZ_SOM_COLUMN = 'SoMs grouped (numbers provided are atom indices)'
AZ_TYPE_COLUMN = 'Exact SoM annotation (1) or extended SoM annotation (0) per group'


def load_cyp_sdf(path):
    """Read one isoform SDF and return molecules with 0-based SoM indices."""
    suppl = Chem.SDMolSupplier(path)
    mols_data = []
    for mol in suppl:
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol)

        som_idxs = []
        if mol.HasProp('PRIMARY_SOM'):
            primary_som = mol.GetProp('PRIMARY_SOM').strip()
            # PRIMARY_SOM is 1-based; convert once, here.
            som_idxs = [int(x) - 1 for x in primary_som.split() if x.isdigit()]

        mols_data.append({
            'mol': mol,
            'smiles': smi,
            'som_idxs': som_idxs,
        })

    return mols_data


def parse_som_groups(som_string):
    """``"['8,9', '12']"`` -> ``[[8, 9], [12]]``."""
    try:
        groups = ast.literal_eval(str(som_string).strip())
    except (ValueError, SyntaxError):
        return []
    if not isinstance(groups, list):
        return []

    result = []
    for group in groups:
        indices = [int(x.strip()) for x in str(group).split(',') if x.strip().isdigit()]
        if indices:
            result.append(indices)
    return result


def parse_annotation_types(annotation_string):
    """``"['1', '0']"`` -> ``[True, False]`` (True = exact SoM)."""
    try:
        types = ast.literal_eval(str(annotation_string).strip())
    except (ValueError, SyntaxError):
        return []
    if not isinstance(types, list):
        return []
    return [bool(int(str(t).strip())) for t in types]


def load_az_csv(csv_path):
    """Read AZ-ExactSOM and keep only exact (atom-resolved) annotations."""
    df = pd.read_csv(csv_path)
    mols_data = []

    for _, row in df.iterrows():
        compound_id = row['Compound ID']
        smiles = row['SMILES']
        som_groups = parse_som_groups(row[AZ_SOM_COLUMN])
        annotation_types = parse_annotation_types(row[AZ_TYPE_COLUMN])

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"[WARN] Failed to parse SMILES for {compound_id}: {smiles}")
            continue

        exact_som_indices = []
        if len(som_groups) == len(annotation_types):
            for group, is_exact in zip(som_groups, annotation_types):
                if is_exact:
                    exact_som_indices.extend(group)

        mols_data.append({
            'mol': mol,
            'smiles': smiles,
            # AZ indices are already 0-based.
            'som_idxs': sorted(set(exact_som_indices)),
            'compound_id': compound_id,
        })

    return mols_data


def read_isoform(base_dir, cyp):
    """Return molecule records for one isoform, or ``None`` if unavailable."""
    csv_path = os.path.join(base_dir, AZ_CSV_NAME)
    sdf_path = os.path.join(base_dir, f'{cyp}.sdf')

    if os.path.exists(csv_path):
        return load_az_csv(csv_path)
    if os.path.exists(sdf_path):
        return load_cyp_sdf(sdf_path)
    return None


def load_multi_cyp(base_dir, cyp_list):
    """Build one graph per (molecule, isoform) pair.

    Every graph carries, in addition to the target-isoform labels ``y``:

    * ``som_annotations`` (N, K) -- SoM labels of the molecule for *all*
      isoforms, used as the soft target of the auxiliary attention loss.
    * ``som_mask`` (N, K) -- 1 where the (molecule, isoform) pair is
      experimentally annotated, 0 otherwise (Eq. 8 in the paper).
    """
    cyp2idx = {c: i for i, c in enumerate(cyp_list)}
    num_cyps = len(cyp_list)

    # Pass 1: collect SoM annotations of every molecule across all isoforms.
    mol_annotations = defaultdict(dict)
    mol_objects = {}

    print("Loading molecules and collecting annotations...")
    for cyp in cyp_list:
        mols_data = read_isoform(base_dir, cyp)
        if mols_data is None:
            print(f"[WARN] No SDF/CSV found for {cyp} under {base_dir}, skipping")
            continue

        print(f"  {cyp}: {len(mols_data)} molecules")
        for mol_data in mols_data:
            smi = mol_data['smiles']
            mol_objects.setdefault(smi, mol_data['mol'])
            mol_annotations[smi][cyp] = mol_data['som_idxs']

    print(f"Total unique molecules: {len(mol_objects)}")

    # Pass 2: build the graphs.
    all_graphs = []
    for cyp in cyp_list:
        mols_data = read_isoform(base_dir, cyp)
        if mols_data is None:
            continue

        for mol_data in tqdm(mols_data, desc=f"Building graphs for {cyp}"):
            mol = mol_data['mol']
            smi = mol_data['smiles']

            data, _ = mol_to_graph(mol, smi, som_indices=mol_data['som_idxs'])
            num_atoms = data.num_nodes

            som_annotations = torch.zeros(num_atoms, num_cyps, dtype=torch.float32)
            som_mask = torch.zeros(num_atoms, num_cyps, dtype=torch.float32)

            for other_idx, other_cyp in enumerate(cyp_list):
                if other_cyp not in mol_annotations[smi]:
                    continue
                som_mask[:, other_idx] = 1.0
                for atom_idx in mol_annotations[smi][other_cyp]:
                    if 0 <= atom_idx < num_atoms:
                        som_annotations[atom_idx, other_idx] = 1.0

            data.som_annotations = som_annotations
            data.som_mask = som_mask
            data.cyp_idx = torch.full((num_atoms,), cyp2idx[cyp], dtype=torch.long)
            data.cyp_name = cyp
            if 'compound_id' in mol_data:
                data.mol_id = mol_data['compound_id']

            all_graphs.append(data)

    all_graphs = [g for g in all_graphs if g.num_nodes > 0]

    print(f"\nLoaded total graphs: {len(all_graphs)}")
    print(f"CYPs: {cyp_list}")

    if len(all_graphs) > 0:
        sample = all_graphs[0]
        print(f"\nSample graph verification:")
        print(f"  num_nodes: {sample.num_nodes}")
        print(f"  som_annotations shape: {tuple(sample.som_annotations.shape)}")
        print(f"  Expected shape: ({sample.num_nodes}, {num_cyps})")
        print(f"  cyp_idx shape: {tuple(sample.cyp_idx.shape)}")
        print(f"  Current CYP: {sample.cyp_name}")
        print(f"  SoM in current CYP: {sample.y.sum().item()}")
        print(f"  SoM annotations sum per CYP: {sample.som_annotations.sum(dim=0).tolist()}")

    return all_graphs


def apply_no_leakage_to_dataloaders(train_set, held_out_set, cyp_list):
    """Mask auxiliary supervision for (molecule, isoform) pairs held out.

    The auxiliary attention loss (Eq. 9) is supervised with the SoM
    annotations of *all* isoforms. A molecule can appear in the training split
    for one isoform and in the test split for another, so the annotations of
    held-out pairs are masked out of ``som_mask``. The atom-level labels ``y``
    are untouched.
    """
    held_out_pairs = {
        (g.smiles, g.cyp_name)
        for g in held_out_set
        if hasattr(g, 'smiles') and hasattr(g, 'cyp_name')
    }

    print(f"\n[Data Leakage Fix with Masking]")

    masked_count = 0
    for g in train_set:
        if not hasattr(g, 'som_mask'):
            g.som_mask = torch.ones(g.num_nodes, len(cyp_list))

        for c_idx, c_name in enumerate(cyp_list):
            if (g.smiles, c_name) in held_out_pairs:
                g.som_mask[:, c_idx] = 0.0
                masked_count += 1

    print(f"Masked {masked_count} (Molecule, CYP) pairs in Train set to prevent leakage.")
    return train_set
