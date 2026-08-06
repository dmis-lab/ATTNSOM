"""Cross-validated training and evaluation of ATTNSOM.

Runs k-fold cross-validation (10 folds in the paper) on either the Zaretzki
dataset or AZ-ExactSOM, and writes per-fold and aggregated metrics under
``--result_dir``.
"""

import os
import json
import math
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import argparse

from dataset import AZ_CSV_NAME, load_multi_cyp, apply_no_leakage_to_dataloaders
from dataset_utils import make_strata_labels, split_train_val_by_index_stratified_cyp
from train import bce_pos_weight, train_one_epoch, evaluate
from metrics import *
from model import ATTNSOM, ENCODERS


ZARETZKI_CYPS = ['1A2', '2A6', '2B6', '2C8', '2C9', '2C19', '2D6', '2E1', '3A4']
# AZ-ExactSOM aggregates human hepatocyte outcomes and carries no isoform
# labels, so it is modelled as a single isoform.
AZ_CYPS = ['3A4']
THRESHOLD = 0.5


def resolve_cyp_list(dataset_dir):
    if os.path.exists(os.path.join(dataset_dir, AZ_CSV_NAME)):
        return AZ_CYPS
    return ZARETZKI_CYPS


def set_wandb(args):
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        wandb.login()
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    wandb.init(
                project = args.project_name ,
                name = run_name,
                config = args
    )

def set_seed(args, deterministic=True):
    seed = args.seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(args, atom_in_dim, edge_dim, cyp_list, device):
    return ATTNSOM(
        atom_in_dim=atom_in_dim,
        edge_dim=edge_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        groups=4,
        mid_K=3,
        dropout=args.dropout,
        cyp_names=cyp_list,
        num_attn_heads=args.num_attn_heads,
        use_attention=not args.no_attention,
        use_film=not args.no_film,
        encoder=args.encoder,
    ).to(device)


def model_config(args, atom_in_dim, edge_dim, cyp_list):
    """Everything needed to rebuild the model at inference time."""
    return {
        'atom_in_dim': atom_in_dim,
        'edge_dim': edge_dim,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'groups': 4,
        'mid_K': 3,
        'dropout': args.dropout,
        'cyp_names': cyp_list,
        'num_attn_heads': args.num_attn_heads,
        'use_attention': not args.no_attention,
        'use_film': not args.no_film,
        'encoder': args.encoder,
    }


def group_results_by_cyp(graphs_subset, all_true, all_probs, cyp_list):
    assert len(graphs_subset) == len(all_true) == len(all_probs)
    buckets = {c: {'true': [], 'probs': []} for c in cyp_list}
    for g, t, p in zip(graphs_subset, all_true, all_probs):
        c = g.cyp_name
        buckets[c]['true'].append(t)
        buckets[c]['probs'].append(p)
    return buckets


def save_molecule_level_predictions(graphs_subset, all_true, all_probs, fold, save_dir, threshold=0.5):
    molecule_predictions = []

    for idx, (graph, true_labels, probs) in enumerate(zip(graphs_subset, all_true, all_probs)):
        preds = (np.array(probs) > threshold).astype(int)

        mol_info = {
            'molecule_id': idx,
            'cyp_name': graph.cyp_name,
            'num_atoms': len(true_labels),
            'true_labels': true_labels,
            'predictions': preds.tolist(),
            'probabilities': probs,
        }

        if hasattr(graph, 'smiles'):
            mol_info['smiles'] = graph.smiles
        if hasattr(graph, 'mol_id'):
            mol_info['mol_id'] = graph.mol_id
        if hasattr(graph, 'name'):
            mol_info['name'] = graph.name

        mol_info['num_true_positives'] = int(np.sum(true_labels))
        mol_info['num_predicted_positives'] = int(np.sum(preds))

        mol_info['exact_match'] = bool(np.array_equal(true_labels, preds))
        mol_info['per_atom_accuracy'] = float(np.mean(np.array(true_labels) == preds))

        molecule_predictions.append(mol_info)

    save_path = os.path.join(save_dir, f'fold{fold}_molecule_predictions.json')
    with open(save_path, 'w') as f:
        json.dump(molecule_predictions, f, indent=2)

    return molecule_predictions


def save_molecule_level_by_cyp(graphs_subset, all_true, all_probs, fold, save_dir, cyp_list, threshold=0.5):
    cyp_molecules = {c: [] for c in cyp_list}

    for idx, (graph, true_labels, probs) in enumerate(zip(graphs_subset, all_true, all_probs)):
        preds = (np.array(probs) > threshold).astype(int)
        cyp = graph.cyp_name

        mol_info = {
            'molecule_id': idx,
            'num_atoms': len(true_labels),
            'true_labels': true_labels,
            'predictions': preds.tolist(),
            'probabilities': probs,
            'num_true_positives': int(np.sum(true_labels)),
            'num_predicted_positives': int(np.sum(preds)),
            'exact_match': bool(np.array_equal(true_labels, preds)),
            'per_atom_accuracy': float(np.mean(np.array(true_labels) == preds)),
        }

        if hasattr(graph, 'smiles'):
            mol_info['smiles'] = graph.smiles
        if hasattr(graph, 'mol_id'):
            mol_info['mol_id'] = graph.mol_id
        if hasattr(graph, 'name'):
            mol_info['name'] = graph.name

        cyp_molecules[cyp].append(mol_info)

    save_path = os.path.join(save_dir, f'fold{fold}_molecule_predictions_by_cyp.json')
    with open(save_path, 'w') as f:
        json.dump(cyp_molecules, f, indent=2)

    return cyp_molecules



def main(args):
    set_seed(args)

    if args.log_wandb: set_wandb(args)

    device = args.device
    print(f"Using device: {device}")

    cyp_list = resolve_cyp_list(args.dataset_dir)
    print(f"CYP isoforms: {cyp_list}")

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir,'folds'), exist_ok=True)

    graphs = load_multi_cyp(args.dataset_dir, cyp_list)
    if len(graphs) == 0:
        raise SystemExit("No data found.")

    n_splits = args.n_splits
    batch_size = args.batch_size
    y_strat = make_strata_labels(graphs)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    atom_in_dim = graphs[0].x.size(-1)
    edge_dim = graphs[0].edge_attr.size(-1)
    config = model_config(args, atom_in_dim, edge_dim, cyp_list)
    with open(os.path.join(args.result_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    if args.save_ckpt:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        with open(os.path.join(args.ckpt_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    fold_summaries_overall = []
    per_cyp_all_folds = {c: [] for c in cyp_list}

    all_folds_results = {
        'true_labels': [],
        'predictions': [],
        'probabilities': [],
        'fold_ids': [],
        'molecule_ids': [],
        'cyp_names': []
    }

    all_molecules = []

    print(f"Running {n_splits}-fold cross-validation")

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.arange(len(graphs)), y_strat), start=1):

        model = build_model(args, atom_in_dim, edge_dim, cyp_list, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        print(f"\n========== Multi-CYP Fold {fold:02d}/{n_splits} ==========")
        tr_idx, val_idx = split_train_val_by_index_stratified_cyp(graphs, trainval_idx, val_ratio=args.inner_val_ratio, seed=args.seed+fold)

        train_set = [graphs[i] for i in tr_idx]
        val_set = [graphs[i] for i in val_idx]
        test_set = [graphs[i] for i in test_idx]


        train_set = apply_no_leakage_to_dataloaders(train_set, test_set, cyp_list)
        train_set = apply_no_leakage_to_dataloaders(train_set, val_set, cyp_list)

        val_set = apply_no_leakage_to_dataloaders(val_set, test_set, cyp_list)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=0)

        best_val = math.inf
        best_state = None
        for epoch in range(1, args.max_epochs+1):
            tr_loss = train_one_epoch(args, model, train_loader, optimizer, device, args.gamma, lambda_main=args.lambda_main, lambda_attn=args.lambda_attn)
            model.eval()
            with torch.no_grad():
                val_total = 0.0
                for vb in val_loader:
                    vb = vb.to(device)
                    logits, _, _ = model(vb)
                    y_true = vb.y.to(logits.dtype)
                    pw = bce_pos_weight(y_true)
                    loss = F.binary_cross_entropy_with_logits(logits, y_true, pos_weight=pw)
                    val_total += loss.item()
                val_loss = val_total / max(1, len(val_loader))
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[Fold {fold:02d}] Epoch {epoch:03d} | train (main {tr_loss['main']:.4f}, attn {tr_loss['attn']:.4f}) | val {val_loss:.4f}")

        # Checkpoint selection: lowest validation loss.
        if best_state is not None:
            print(f"[Fold {fold:02d}] best val loss: {best_val:.4f}")
            model.load_state_dict(best_state)
            if args.save_ckpt:
                torch.save(best_state, os.path.join(args.ckpt_dir, f'fold{fold}_best.pt'))

        all_true, all_probs = evaluate(model, test_loader, device)

        m = calc_threshold_metrics(all_true, all_probs, THRESHOLD)
        t1 = topk_accuracy(all_true, all_probs, 1)
        t2 = topk_accuracy(all_true, all_probs, 2)
        t3 = topk_accuracy(all_true, all_probs, 3)
        paa = per_atom_accuracy(all_true, all_probs, THRESHOLD)
        mem = molecule_exact_match(all_true, all_probs, THRESHOLD)

        fold_summary_overall = {
            'fold': fold,
            'precision': m['precision'], 'recall': m['recall'], 'f1': m['f1_binary'], 'mcc': m['mcc'],
            'top1': t1, 'top2': t2, 'top3': t3, 'per_atom_acc': paa, 'molecule_exact': mem,
            'n_train': len(train_set), 'n_val': len(val_set), 'n_test': len(test_set)
        }
        fold_summaries_overall.append(fold_summary_overall)

        buckets = group_results_by_cyp(test_set, all_true, all_probs, cyp_list)
        per_cyp_fold = {}
        for c in cyp_list:
            T = buckets[c]['true']; P = buckets[c]['probs']
            if len(T) == 0:
                continue
            mm  = calc_threshold_metrics(T, P, THRESHOLD)
            tt1 = topk_accuracy(T, P, 1)
            tt2 = topk_accuracy(T, P, 2)
            tt3 = topk_accuracy(T, P, 3)
            ppaa = per_atom_accuracy(T, P, THRESHOLD)
            mmem = molecule_exact_match(T, P, THRESHOLD)
            per_cyp_metrics = {
                'fold': fold, 'cyp': c, 'n_mols': len(T),
                'precision': mm['precision'], 'recall': mm['recall'], 'f1': mm['f1_binary'], 'mcc': mm['mcc'],
                'top1': tt1, 'top2': tt2, 'top3': tt3, 'per_atom_acc': ppaa, 'molecule_exact': mmem,

            }
            per_cyp_fold[c] = per_cyp_metrics
            per_cyp_all_folds[c].append(per_cyp_metrics)

        with open(os.path.join(args.result_dir, 'folds', f'multi_cyp_fold{fold}_per_cyp.json'), 'w') as f:
            json.dump(per_cyp_fold, f, indent=4)

        all_true_list = []
        all_probs_list = []
        for true_labels, probs in zip(all_true, all_probs):
            if isinstance(true_labels, torch.Tensor):
                true_labels = true_labels.cpu().tolist()
            if isinstance(probs, torch.Tensor):
                probs = probs.cpu().tolist()
            all_true_list.append(true_labels)
            all_probs_list.append(probs)

        all_preds = []
        for probs in all_probs_list:
            preds = (np.array(probs) > THRESHOLD).astype(int).tolist()
            all_preds.append(preds)


        fold_results = {
            'true_labels': all_true_list,
            'predictions': all_preds,
            'probabilities': all_probs_list,
            'test_indices': test_idx.tolist()
        }

        np.savez(
            os.path.join(args.result_dir, 'folds', f'fold{fold}_predictions.npz'),
            true_labels=np.array(all_true_list, dtype=object),
            predictions=np.array(all_preds, dtype=object),
            probabilities=np.array(all_probs_list, dtype=object),
            test_indices=test_idx
        )

        with open(os.path.join(args.result_dir, 'folds', f'fold{fold}_predictions.json'), 'w') as f:
            json.dump(fold_results, f, indent=4)

        molecule_preds = save_molecule_level_predictions(
            test_set, all_true_list, all_probs_list, fold,
            os.path.join(args.result_dir, 'folds'),
            threshold=THRESHOLD
        )

        save_molecule_level_by_cyp(
            test_set, all_true_list, all_probs_list, fold,
            os.path.join(args.result_dir, 'folds'), cyp_list,
            threshold=THRESHOLD
        )

        for mol_idx, (graph, true_labels, probs, preds) in enumerate(zip(test_set, all_true_list, all_probs_list, all_preds)):
            all_folds_results['true_labels'].extend(true_labels)
            all_folds_results['predictions'].extend(preds)
            all_folds_results['probabilities'].extend(probs)
            all_folds_results['fold_ids'].extend([fold] * len(true_labels))
            all_folds_results['molecule_ids'].extend([mol_idx] * len(true_labels))
            all_folds_results['cyp_names'].extend([graph.cyp_name] * len(true_labels))

        for mol_pred in molecule_preds:
            mol_pred['fold'] = fold
            all_molecules.append(mol_pred)


    np.savez(
        os.path.join(args.result_dir, 'all_folds_predictions.npz'),
        true_labels=np.array(all_folds_results['true_labels']),
        predictions=np.array(all_folds_results['predictions']),
        probabilities=np.array(all_folds_results['probabilities']),
        fold_ids=np.array(all_folds_results['fold_ids']),
        molecule_ids=np.array(all_folds_results['molecule_ids']),
        cyp_names=np.array(all_folds_results['cyp_names'])
    )

    with open(os.path.join(args.result_dir, 'all_folds_predictions.json'), 'w') as f:
        json.dump(all_folds_results, f, indent=4)

    with open(os.path.join(args.result_dir, 'all_molecules_predictions.json'), 'w') as f:
        json.dump(all_molecules, f, indent=2)

    all_cyp_molecules = {c: [] for c in cyp_list}
    for mol in all_molecules:
        all_cyp_molecules[mol['cyp_name']].append(mol)

    with open(os.path.join(args.result_dir, 'all_molecules_by_cyp.json'), 'w') as f:
        json.dump(all_cyp_molecules, f, indent=2)

    keys = ['precision','recall','f1','top1','top2','top3','per_atom_acc','molecule_exact', 'mcc']
    agg_overall = {f'{k}_mean': float(np.mean([fs[k] for fs in fold_summaries_overall])) for k in keys}
    agg_overall.update({f'{k}_std': float(np.std([fs[k] for fs in fold_summaries_overall], ddof=1)) for k in keys})
    agg_overall['n_splits'] = n_splits

    agg_per_cyp = {}
    for c in cyp_list:
        rows = per_cyp_all_folds[c]
        if not rows:
            continue

        agg = {f'{k}_mean': float(np.mean([r[k] for r in rows])) for k in keys}
        agg.update({f'{k}_std': float(np.std([r[k] for r in rows], ddof=1)) for k in keys})
        agg['n_folds_present'] = len(rows)
        agg_per_cyp[c] = agg

    with open(os.path.join(args.result_dir, 'summary.json'), 'w') as f:
        json.dump({'overall': agg_overall, 'per_cyp': agg_per_cyp}, f, indent=4)

    print(f"\n=== Multi-CYP {n_splits}-fold CV aggregate (overall) ===")
    for k in keys:
        print(f"{k:>16}: {agg_overall[f'{k}_mean']:.4f} ± {agg_overall[f'{k}_std']:.4f}")

    print(f"\n=== Multi-CYP {n_splits}-fold CV aggregate (per CYP) ===")
    for c, agg in agg_per_cyp.items():
        line = ", ".join([f"{k.replace('_mean','')}: {agg[k+'_mean']:.4f}±{agg[k + '_std']:.4f}" for k in keys if k+'_mean' in agg])
        print(f"{c}: {line}")

    print("\n=== Saved Files ===")
    print(f"- Atom-level predictions: {args.result_dir}/all_folds_predictions.npz/.json")
    print(f"- Molecule-level predictions: {args.result_dir}/all_molecules_predictions.json")
    print(f"- CYP-grouped molecules: {args.result_dir}/all_molecules_by_cyp.json")
    print(f"- Per-fold results: {args.result_dir}/folds/")
    if args.save_ckpt:
        print(f"- Checkpoints: {args.ckpt_dir}/fold*_best.pt")

    if args.log_wandb:
        logging = {}
        for c, agg in agg_per_cyp.items():
            for k in keys:
                logging[f'{c}_{k}_mean'] = round(agg[f'{k}_mean'], 4)
                logging[f'{c}_{k}_std'] = round(agg[f'{k}_std'], 4)

        logging.update(agg_overall)
        wandb.log(logging)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate ATTNSOM with k-fold CV.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_dir', default='./cyp_dataset',
                        help="Zaretzki: directory of <isoform>.sdf. AZ-ExactSOM: ./cyp_dataset/az")
    parser.add_argument('--result_dir', default='results')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--project_name', default="ATTNSOM", type=str)
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--log_wandb', '-w', default=False, action='store_true')

    parser.add_argument('--n_splits', default=10, type=int)
    parser.add_argument('--inner_val_ratio', '-ivr', default=0.05, type=float)

    parser.add_argument('--max_epochs', '-e', default=50, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--loss', default='focal', type=str, choices=['bce', 'focal'])
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--gamma', default=1, type=float, help="Focal loss gamma")
    parser.add_argument('--pos_weight', default=1.5, type=float,
                        help="Positive-class weight of the main loss")

    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--encoder', default='graphcliff', type=str, choices=ENCODERS,
                        help="Graph backbone (Table 3 encoder ablations)")

    parser.add_argument('--lambda_main', '-lm', default=1, type=float)
    parser.add_argument('--lambda_attn', '-la', default=0.5, type=float)
    parser.add_argument('--attn_loss_type', '-at', default='bce', type=str, choices=['bce', 'kl'])
    parser.add_argument('--num_attn_heads', '-nh', default=4, type=int)

    parser.add_argument('--no_attention', '-na', default=False, action='store_true',
                        help="Ablation: drop the cross-isoform attention module")
    parser.add_argument('--no_film', '-nf', default=False, action='store_true',
                        help="Ablation: drop the FiLM molecule conditioning")

    parser.add_argument('--save_ckpt', default=False, action='store_true',
                        help="Save the best checkpoint of every fold (needed by inference.py)")
    parser.add_argument('--ckpt_dir', default='checkpoint/attnsom')

    args = parser.parse_args()

    if args.log_wandb:
        import wandb

    main(args)
