"""Predict sites of metabolism for arbitrary molecules with a trained ATTNSOM.

Predictions are averaged over the checkpoints of the cross-validation folds,
which is the ensembling scheme used for the case studies in the paper.

Example:
    python inference.py --ckpt_dir checkpoint/attnsom \\
        --smiles "CC(C)(C)c1cc(cc(c1O)C(C)(C)C)..." --save_attention
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch

from dataset_utils import mol_to_graph
from model import ATTNSOM

THRESHOLD = 0.5


def read_smiles(args):
    smiles = []
    if args.smiles:
        smiles.extend(args.smiles)
    if args.smiles_file:
        with open(args.smiles_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    smiles.append(line.split()[0])
    if not smiles:
        raise SystemExit("Provide molecules with --smiles and/or --smiles_file.")
    return smiles


def load_config(ckpt_dir, result_dir=None):
    for directory in (ckpt_dir, result_dir):
        if not directory:
            continue
        path = os.path.join(directory, 'config.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise SystemExit(
        f"No config.json found in {ckpt_dir}. Train with --save_ckpt to create one."
    )


def load_models(ckpt_dir, config, device):
    paths = sorted(
        glob.glob(os.path.join(ckpt_dir, 'fold*_best.pt')),
        key=lambda p: int(''.join(ch for ch in os.path.basename(p) if ch.isdigit()))
    )
    if not paths:
        raise SystemExit(f"No fold*_best.pt checkpoints found in {ckpt_dir}.")

    models = []
    for path in paths:
        model = ATTNSOM(**config).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append((os.path.basename(path), model))
    print(f"Loaded {len(models)} fold checkpoints from {ckpt_dir}")
    return models


def build_batch(mol, smiles, cyp_list, device):
    """One graph per isoform, batched together."""
    graphs = []
    for cyp_idx in range(len(cyp_list)):
        data, _ = mol_to_graph(mol, smiles, som_indices=[])
        data.cyp_idx = torch.full((data.num_nodes,), cyp_idx, dtype=torch.long)
        graphs.append(data)
    return Batch.from_data_list(graphs).to(device)


@torch.no_grad()
def predict_molecule(models, mol, smiles, cyp_list, device):
    """Return (num_isoforms, num_atoms) probabilities and attention maps."""
    batch = build_batch(mol, smiles, cyp_list, device)
    num_atoms = mol.GetNumAtoms()
    K = len(cyp_list)

    probs_per_fold = []
    attn_per_fold = []
    for _, model in models:
        logits, _, attn = model(batch)
        probs = torch.sigmoid(logits).view(K, num_atoms)
        probs_per_fold.append(probs.cpu().numpy())
        if attn is not None:
            # Attention does not depend on the target isoform, so take the
            # first replica: (num_atoms, K).
            attn_per_fold.append(attn.view(K, num_atoms, K)[0].cpu().numpy())

    probs = np.mean(probs_per_fold, axis=0)
    attn = np.mean(attn_per_fold, axis=0) if attn_per_fold else None
    return probs, attn


def plot_attention(attn, cyp_list, symbols, out_path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.0 * len(cyp_list) + 2, 0.28 * len(symbols) + 2))
    im = ax.imshow(attn, aspect='auto', cmap='YlOrRd', vmin=0.0)
    ax.set_xticks(range(len(cyp_list)))
    ax.set_xticklabels([f'CYP{c}' for c in cyp_list], rotation=45, ha='right')
    ax.set_yticks(range(len(symbols)))
    ax.set_yticklabels([f'{i}:{s}' for i, s in enumerate(symbols)], fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label='attention weight')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(args):
    device = args.device
    config = load_config(args.ckpt_dir, args.result_dir)
    cyp_list = config['cyp_names']
    models = load_models(args.ckpt_dir, config, device)

    if args.cyp:
        unknown = [c for c in args.cyp if c not in cyp_list]
        if unknown:
            raise SystemExit(f"Unknown isoform(s) {unknown}; available: {cyp_list}")
        selected = args.cyp
    else:
        selected = cyp_list

    os.makedirs(args.output_dir, exist_ok=True)
    smiles_list = read_smiles(args)

    results = []
    rows = []
    for mol_i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"[WARN] Could not parse SMILES: {smi}")
            continue

        canonical = Chem.MolToSmiles(mol)
        probs, attn = predict_molecule(models, mol, canonical, cyp_list, device)
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]

        entry = {
            'input_smiles': smi,
            'canonical_smiles': canonical,
            'num_atoms': mol.GetNumAtoms(),
            'atom_symbols': symbols,
            'predictions': {},
        }

        for cyp in selected:
            k = cyp_list.index(cyp)
            p = probs[k]
            order = np.argsort(-p)
            entry['predictions'][cyp] = {
                'probabilities': p.round(6).tolist(),
                'predicted_som_atoms': np.where(p > args.threshold)[0].tolist(),
                'top3_atoms': order[:3].tolist(),
            }
            for atom_idx in range(len(p)):
                rows.append({
                    'molecule_index': mol_i,
                    'smiles': canonical,
                    'cyp': cyp,
                    'atom_index': atom_idx,
                    'atom_symbol': symbols[atom_idx],
                    'probability': float(p[atom_idx]),
                    'predicted_som': int(p[atom_idx] > args.threshold),
                })

        if attn is not None and args.save_attention:
            np.save(os.path.join(args.output_dir, f'mol{mol_i}_attention.npy'), attn)
            plot_attention(attn, cyp_list, symbols,
                           os.path.join(args.output_dir, f'mol{mol_i}_attention.png'),
                           f'Atom-CYP attention\n{canonical}')
            entry['attention_map'] = attn.round(6).tolist()

        results.append(entry)

        print(f"\n[{mol_i}] {canonical}")
        for cyp in selected:
            top = entry['predictions'][cyp]['top3_atoms']
            k = cyp_list.index(cyp)
            desc = ", ".join(f"{i}:{symbols[i]}({probs[k][i]:.3f})" for i in top)
            print(f"  CYP{cyp:<5} top-3: {desc}")

    json_path = os.path.join(args.output_dir, 'predictions.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(args.output_dir, 'predictions.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {json_path} and {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ATTNSOM inference on new molecules.")
    parser.add_argument('--ckpt_dir', default='checkpoint/attnsom',
                        help="Directory with fold*_best.pt and config.json")
    parser.add_argument('--result_dir', default=None,
                        help="Fallback directory to read config.json from")
    parser.add_argument('--smiles', nargs='+', default=None)
    parser.add_argument('--smiles_file', default=None,
                        help="Text file with one SMILES per line")
    parser.add_argument('--cyp', nargs='+', default=None,
                        help="Restrict output to these isoforms, e.g. --cyp 3A4 2D6")
    parser.add_argument('--output_dir', default='inference_results')
    parser.add_argument('--threshold', type=float, default=THRESHOLD)
    parser.add_argument('--save_attention', action='store_true',
                        help="Store the atom-isoform attention map and a heatmap")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    main(parser.parse_args())
