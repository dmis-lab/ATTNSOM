"""Cross-isoform SoM pattern similarity (Figures 1 and 5).

* ``--source dataset``  -- pairwise Jaccard similarity of the *annotated* SoM
  atoms across isoforms (Figure 1).
* ``--source predicted`` -- the same statistic computed from ATTNSOM's
  cross-validated predictions (Figure 5).

Running both also reports the cophenetic correlation between the two
dendrograms.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import squareform

from dataset import AZ_CSV_NAME, read_isoform

ZARETZKI_CYPS = ['1A2', '2A6', '2B6', '2C8', '2C9', '2C19', '2D6', '2E1', '3A4']


def dataset_som_sets(dataset_dir, cyp_list):
    """``{isoform: {smiles: set(som atom indices)}}`` from the annotations."""
    from rdkit import Chem

    som_sets = {}
    for cyp in cyp_list:
        mols_data = read_isoform(dataset_dir, cyp)
        if mols_data is None:
            print(f"[WARN] no data for {cyp}")
            continue
        per_mol = {}
        for mol_data in mols_data:
            smi = Chem.MolToSmiles(mol_data['mol'])
            per_mol[smi] = set(mol_data['som_idxs'])
        som_sets[cyp] = per_mol
    return som_sets


def predicted_som_sets(result_dir, threshold=0.5):
    """``{isoform: {smiles: set(predicted som atom indices)}}`` from CV output."""
    path = os.path.join(result_dir, 'all_molecules_predictions.json')
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found. Run main.py first so that predictions are written."
        )
    with open(path) as f:
        molecules = json.load(f)

    som_sets = {}
    for mol in molecules:
        cyp = mol['cyp_name']
        smi = mol.get('smiles')
        if smi is None:
            continue
        probs = np.asarray(mol['probabilities'])
        som_sets.setdefault(cyp, {})[smi] = set(np.where(probs > threshold)[0].tolist())
    return som_sets


def pairwise_jaccard(som_sets, cyp_list, min_shared=1):
    """Mean Jaccard similarity of SoM atom sets over co-annotated molecules."""
    K = len(cyp_list)
    sim = np.eye(K)
    counts = np.zeros((K, K), dtype=int)

    for i in range(K):
        for j in range(i + 1, K):
            a = som_sets.get(cyp_list[i], {})
            b = som_sets.get(cyp_list[j], {})
            shared = set(a) & set(b)
            scores = []
            for smi in shared:
                union = a[smi] | b[smi]
                if not union:
                    continue
                scores.append(len(a[smi] & b[smi]) / len(union))
            counts[i, j] = counts[j, i] = len(scores)
            value = float(np.mean(scores)) if len(scores) >= min_shared else np.nan
            sim[i, j] = sim[j, i] = value
    return sim, counts


def plot_similarity(sim, cyp_list, out_path, title):
    """Heatmap with a dendrogram, ordered by hierarchical clustering."""
    dist = 1.0 - np.nan_to_num(sim, nan=0.0)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method='average')
    order = dendrogram(Z, no_plot=True)['leaves']

    fig = plt.figure(figsize=(7.5, 6.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[4, 0.2],
                            hspace=0.05, wspace=0.05)

    ax_dendro = fig.add_subplot(grid[0, 0])
    dendrogram(Z, ax=ax_dendro, labels=[f'CYP{c}' for c in cyp_list],
               color_threshold=0, above_threshold_color='0.3')
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    for spine in ax_dendro.spines.values():
        spine.set_visible(False)
    ax_dendro.set_title(title)

    ax = fig.add_subplot(grid[1, 0])
    ordered = sim[np.ix_(order, order)]
    im = ax.imshow(ordered, cmap='viridis')
    labels = [f'CYP{cyp_list[i]}' for i in order]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = ordered[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f'{value:.3f}', ha='center', va='center', fontsize=7,
                    color='white' if value < 0.75 * np.nanmax(ordered) else 'black')

    cax = fig.add_subplot(grid[1, 1])
    fig.colorbar(im, cax=cax, label='Jaccard similarity')

    fig.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    return Z


def report(sim, cyp_list, counts, name):
    off = sim[~np.eye(len(cyp_list), dtype=bool)]
    off = off[~np.isnan(off)]
    print(f"\n=== {name} SoM pattern similarity ===")
    print(f"pairs: {len(off) // 2}, min: {off.min():.3f}, max: {off.max():.3f}, "
          f"mean: {off.mean():.3f}")
    print(f"co-annotated molecules per pair: {counts[np.triu_indices(len(cyp_list), 1)]}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    cyp_list = ZARETZKI_CYPS
    if os.path.exists(os.path.join(args.dataset_dir, AZ_CSV_NAME)):
        raise SystemExit("Cross-isoform similarity requires the isoform-resolved "
                         "Zaretzki dataset, not AZ-ExactSOM.")

    linkages = {}
    matrices = {}

    if args.source in ('dataset', 'both'):
        som_sets = dataset_som_sets(args.dataset_dir, cyp_list)
        sim, counts = pairwise_jaccard(som_sets, cyp_list)
        report(sim, cyp_list, counts, 'Annotated (dataset)')
        out = os.path.join(args.output_dir, 'similarity_dataset.png')
        linkages['dataset'] = plot_similarity(
            sim, cyp_list, out, 'Annotated SoM pattern similarity (Zaretzki)')
        matrices['dataset'] = sim
        print(f"saved {out}")

    if args.source in ('predicted', 'both'):
        som_sets = predicted_som_sets(args.result_dir, args.threshold)
        sim, counts = pairwise_jaccard(som_sets, cyp_list)
        report(sim, cyp_list, counts, 'Predicted (ATTNSOM)')
        out = os.path.join(args.output_dir, 'similarity_predicted.png')
        linkages['predicted'] = plot_similarity(
            sim, cyp_list, out, 'Predicted SoM pattern similarity (ATTNSOM)')
        matrices['predicted'] = sim
        print(f"saved {out}")

    if len(linkages) == 2:
        d_data = squareform(1.0 - np.nan_to_num(matrices['dataset']), checks=False)
        d_pred = squareform(1.0 - np.nan_to_num(matrices['predicted']), checks=False)
        c_data = cophenet(linkages['dataset'], d_data)[1]
        c_pred = cophenet(linkages['predicted'], d_pred)[1]
        r = np.corrcoef(c_data, c_pred)[0, 1]
        print(f"\nCophenetic correlation between the two dendrograms: r = {r:.3f}")

    np.savez(os.path.join(args.output_dir, 'similarity_matrices.npz'),
             cyp_list=np.array(cyp_list), **matrices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cross-isoform SoM similarity analysis.")
    parser.add_argument('--source', default='both', choices=['dataset', 'predicted', 'both'])
    parser.add_argument('--dataset_dir', default='./cyp_dataset')
    parser.add_argument('--result_dir', default='results',
                        help="Directory produced by main.py (for --source predicted)")
    parser.add_argument('--output_dir', default='figures')
    parser.add_argument('--threshold', type=float, default=0.5)
    main(parser.parse_args())
