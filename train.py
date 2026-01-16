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


def compute_attention_alignment_loss(attn_weights, som_annotations, has_som_mask):
    effective_som = som_annotations * has_som_mask
    som_sums = effective_som.sum(dim=-1, keepdim=True) 
    
    target_dist = effective_som / (som_sums + 1e-10)

    log_attn = torch.log(attn_weights + 1e-10)
    kl_loss_elements = F.kl_div(log_attn, target_dist, reduction='none')

    masked_kl_loss = (kl_loss_elements * has_som_mask).sum() / (has_som_mask.sum() + 1e-10)
    
    return masked_kl_loss 


def compute_attention_alignment_loss_bce(attn_weights, som_annotations, has_som_mask):
    effective_som = som_annotations * has_som_mask
    som_sums = effective_som.sum(dim=-1, keepdim=True)
    
    target_dist = effective_som / (som_sums + 1e-10) 

    bce_elements = F.binary_cross_entropy(attn_weights, target_dist, reduction='none')

    loss = (bce_elements * has_som_mask).sum() / (has_som_mask.sum() + 1e-10)

    return loss

def compute_attention_entropy_regularization(attn_weights, y_true, som_annotations=None):
    N, num_cyps = attn_weights.shape

    entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-10), dim=-1)
    
    if som_annotations is not None:
        som_sum = som_annotations.sum(dim=-1)
        has_any_som = (som_sum > 0).float()
        
        som_mask = (y_true > 0.5).float() 
        non_som_mask = (y_true <= 0.5).float() 
        
        som_mask = som_mask * has_any_som
        non_som_mask = non_som_mask * has_any_som
    else:
        som_mask = (y_true > 0.5).float()
        non_som_mask = (y_true <= 0.5).float()
    
    loss = attn_weights.new_tensor(0.0)
    count = 0
    
    if som_mask.sum() > 0:
        loss_som = (entropy * som_mask).sum() / (som_mask.sum() + 1e-10)
        loss = loss + loss_som
        count += 1
    
    if non_som_mask.sum() > 0:
        max_entropy = torch.log(torch.tensor(num_cyps, dtype=torch.float, device=attn_weights.device))
        loss_non_som = ((max_entropy - entropy) * non_som_mask).sum() / (non_som_mask.sum() + 1e-10)
        loss = loss + loss_non_som
        count += 1
    
    return loss / max(count, 1)


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
    stats = {k: 0.0 for k in ['total', 'main', 'somcon', 'supcon', 'attn']}
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits, node_repr, attn_weights = get_logits_and_repr(model, batch)
        y_true = batch.y.to(logits.dtype)
        cyp_idx = batch.cyp_idx

        pw = torch.tensor(1.5, device=device)
        if args.loss == 'focal':
            loss_main = focal_bce_with_logits(logits, y_true, pos_weight=pw, gamma=gamma)
        else:
            loss_main = F.binary_cross_entropy_with_logits(logits, y_true, pos_weight=pw)

        if attn_weights is not None:
            loss_attn = compute_attention_alignment_loss_bce(
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

    n = len(loader)
    return {k: v / n for k, v in stats.items()}


def evaluate(model, loader, device):
    model.eval()
    all_true = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            cyp_idx = batch.cyp_idx 
            try:
                out = model.inference(batch)
                logits = out[0] if isinstance(out, tuple) else out
            except:
                out = model(batch)
                logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits)
            
            if hasattr(batch, 'ptr') and batch.ptr is not None:
                for i in range(batch.ptr.numel() - 1):
                    s, e = int(batch.ptr[i].item()), int(batch.ptr[i+1].item())
                    all_probs.append(probs[s:e].detach().cpu())
                    all_true.append(batch.y[s:e].long().detach().cpu().tolist())
            else:
                bvec = batch.batch
                B = int(bvec.max().item()) + 1 if bvec.numel() > 0 else 0
                for i in range(B):
                    idx = (bvec == i).nonzero(as_tuple=False).view(-1)
                    all_probs.append(probs[idx].detach().cpu())
                    all_true.append(batch.y[idx].long().detach().cpu().tolist())
    
    return all_true, all_probs