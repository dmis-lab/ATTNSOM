import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef


def calc_threshold_metrics(all_true, all_probs, threshold=0.5):
    y_true_flat, y_pred_flat = [], []
    for t, p in zip(all_true, all_probs):
        k = len(t)
        pred = (p[:k] > threshold).long().tolist()
        y_true_flat.extend(t)
        y_pred_flat.extend(pred)
    
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1_bin = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    f1_micro = f1_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
    f1_macro = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true_flat, y_pred_flat)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_binary': f1_bin,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'mcc': mcc,
        'threshold': threshold,
    }


def topk_accuracy(all_true, all_probs, k=1):
    correct = 0
    total = 0
    for t, p in zip(all_true, all_probs):
        k_atoms = len(t)
        if sum(t) == 0:
            continue
        k_eff = min(k, k_atoms)
        
        if isinstance(p, torch.Tensor):
            p = p[:k_atoms]
        else:
            p = torch.tensor(p[:k_atoms])
        
        vals, idxs = torch.topk(p, k_eff)
        idxs = set(idxs.tolist())
        true_idxs = {i for i, v in enumerate(t) if v == 1}
        
        if len(idxs.intersection(true_idxs)) > 0:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def per_atom_accuracy(all_true, all_probs, threshold=0.5):
    correct = 0
    total = 0
    for t, p in zip(all_true, all_probs):
        k = len(t)
        if isinstance(p, torch.Tensor):
            pred = (p[:k] > threshold).long().tolist()
        else:
            pred = (np.array(p[:k]) > threshold).astype(int).tolist()
        
        for a, b in zip(t, pred):
            correct += int(a == b)
            total += 1
    
    return correct / total if total > 0 else 0.0


def molecule_exact_match(all_true, all_probs, threshold=0.5):
    exact = 0
    total = 0
    for t, p in zip(all_true, all_probs):
        k = len(t)
        if isinstance(p, torch.Tensor):
            pred = (p[:k] > threshold).long().tolist()
        else:
            pred = (np.array(p[:k]) > threshold).astype(int).tolist()
        
        exact += int(pred == t)
        total += 1
    
    return exact / total if total > 0 else 0.0


def per_atom_accuracy_cyp_specific(graphs_subset, all_true, all_probs, thresholds):
    correct = 0
    total = 0
    
    for graph, t, p in zip(graphs_subset, all_true, all_probs):
        cyp_name = graph.cyp_name
        thresh = thresholds[cyp_name]
        k = len(t)
        
        if isinstance(p, torch.Tensor):
            pred = (p[:k] > thresh).long().tolist()
        else:
            pred = (np.array(p[:k]) > thresh).astype(int).tolist()
        
        for a, b in zip(t, pred):
            correct += int(a == b)
            total += 1
    
    return correct / total if total > 0 else 0.0


def topk_accuracy_cyp_specific(graphs_subset, all_true, all_probs, thresholds, k=1):
    correct = 0
    total = 0
    
    for graph, t, p in zip(graphs_subset, all_true, all_probs):
        k_atoms = len(t)
        
        if sum(t) == 0:
            continue
        
        k_eff = min(k, k_atoms)
        
        if isinstance(p, torch.Tensor):
            p = p[:k_atoms]
        else:
            p = torch.tensor(p[:k_atoms])
        
        vals, idxs = torch.topk(p, k_eff)
        idxs = set(idxs.tolist())
        true_idxs = {i for i, v in enumerate(t) if v == 1}
        
        if len(idxs.intersection(true_idxs)) > 0:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def molecule_exact_match_cyp_specific(graphs_subset, all_true, all_probs, thresholds):
    exact = 0
    total = 0
    
    for graph, t, p in zip(graphs_subset, all_true, all_probs):
        cyp_name = graph.cyp_name
        thresh = thresholds[cyp_name]
        k = len(t)
        
        if isinstance(p, torch.Tensor):
            pred = (p[:k] > thresh).long().tolist()
        else:
            pred = (np.array(p[:k]) > thresh).astype(int).tolist()
        
        exact += int(pred == t)
        total += 1
    
    return exact / total if total > 0 else 0.0