"""Losses and train/eval loops.

The objective is ``L = lambda_main * L_main + lambda_attn * L_attn`` (Eq. 6):

* ``L_main`` -- Focal loss over atom-wise logits for the target isoform,
* ``L_attn`` -- masked soft-label BCE that aligns the atom-to-isoform attention
  with the experimentally annotated SoMs (Eq. 7-9).
"""

import torch
import torch.nn.functional as F


def bce_pos_weight(y_true):
    pos = (y_true > 0.5).float()
    n_pos = pos.sum().clamp(min=1.0)
    n_neg = (1.0 - pos).sum().clamp(min=1.0)
    return n_neg / n_pos

def focal_bce_with_logits(logits, targets, pos_weight=None, gamma=2.0, eps=1e-8):
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=pos_weight
    )

    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1 - p).clamp(min=eps, max=1-eps)

    focal_factor = (1 - pt) ** gamma

    return (focal_factor * bce).mean()


def attention_soft_targets(som_annotations, has_som_mask):
    """Soft target distribution of Eq. 7.

    Probability mass 1/|P_i| is spread uniformly over the isoforms for which
    atom ``i`` is annotated as a SoM; annotated-but-negative isoforms get 0.
    """
    effective_som = som_annotations * has_som_mask
    som_sums = effective_som.sum(dim=-1, keepdim=True)
    return effective_som / (som_sums + 1e-10)


def compute_attention_alignment_loss_bce(attn_weights, som_annotations, has_som_mask):
    """Masked soft-label BCE between attention weights and SoM targets (Eq. 9)."""
    target_dist = attention_soft_targets(som_annotations, has_som_mask)

    bce_elements = F.binary_cross_entropy(attn_weights, target_dist, reduction='none')

    return (bce_elements * has_som_mask).sum() / (has_som_mask.sum() + 1e-10)


def compute_attention_alignment_loss_kl(attn_weights, som_annotations, has_som_mask):
    """KL variant of the auxiliary loss (``--attn_loss_type kl``)."""
    target_dist = attention_soft_targets(som_annotations, has_som_mask)

    log_attn = torch.log(attn_weights + 1e-10)
    kl_loss_elements = F.kl_div(log_attn, target_dist, reduction='none')

    return (kl_loss_elements * has_som_mask).sum() / (has_som_mask.sum() + 1e-10)


ATTENTION_LOSSES = {
    'bce': compute_attention_alignment_loss_bce,
    'kl': compute_attention_alignment_loss_kl,
}


def get_logits_and_repr(model, batch):
    out = model(batch)

    if len(out) == 3:
        logits, node_repr, attn_weights = out
    elif len(out) == 2:
        logits, node_repr = out
        attn_weights = None
    else:
        logits = out
        node_repr = None
        attn_weights = None

    return logits, node_repr, attn_weights



def train_one_epoch(args, model, loader, optimizer, device, gamma,
                    lambda_main=1.0, lambda_attn=1.0):
    model.train()
    stats = {k: 0.0 for k in ['total', 'main', 'attn']}
    attn_loss_fn = ATTENTION_LOSSES[args.attn_loss_type]

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits, node_repr, attn_weights = get_logits_and_repr(model, batch)
        y_true = batch.y.to(logits.dtype)

        pw = torch.tensor(args.pos_weight, device=device)
        if args.loss == 'focal':
            loss_main = focal_bce_with_logits(logits, y_true, pos_weight=pw, gamma=gamma)
        else:
            loss_main = F.binary_cross_entropy_with_logits(logits, y_true, pos_weight=pw)

        if attn_weights is not None:
            loss_attn = attn_loss_fn(
                attn_weights, batch.som_annotations, batch.som_mask
            )
        else:
            loss_attn = torch.tensor(0.0, device=device)

        loss = (lambda_main * loss_main +
                lambda_attn * loss_attn)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        stats['total'] += loss.item()
        stats['main'] += loss_main.item()
        stats['attn'] += loss_attn.item()

    n = max(1, len(loader))
    return {k: v / n for k, v in stats.items()}


def evaluate(model, loader, device, return_attention=False):
    """Return per-molecule label lists and predicted probabilities."""
    model.eval()
    all_true = []
    all_probs = []
    all_attn = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _, attn_weights = get_logits_and_repr(model, batch)
            probs = torch.sigmoid(logits)

            if hasattr(batch, 'ptr') and batch.ptr is not None:
                spans = [(int(batch.ptr[i].item()), int(batch.ptr[i + 1].item()))
                         for i in range(batch.ptr.numel() - 1)]
            else:
                bvec = batch.batch
                B = int(bvec.max().item()) + 1 if bvec.numel() > 0 else 0
                spans = []
                for i in range(B):
                    idx = (bvec == i).nonzero(as_tuple=False).view(-1)
                    spans.append((int(idx.min().item()), int(idx.max().item()) + 1))

            for s, e in spans:
                all_probs.append(probs[s:e].detach().cpu())
                all_true.append(batch.y[s:e].long().detach().cpu().tolist())
                if return_attention:
                    all_attn.append(
                        attn_weights[s:e].detach().cpu() if attn_weights is not None else None
                    )

    if return_attention:
        return all_true, all_probs, all_attn
    return all_true, all_probs
