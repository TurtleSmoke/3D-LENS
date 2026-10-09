"""
Identity losses adapted from fast-reid (https://github.com/JDAI-CV/fast-reid), by liaoxingyu.
"""

import torch
import torch.nn.functional as F


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    return dist.clamp(min=1e-12).sqrt()


def triplet_loss(embedding, targets):
    """Batch-hard triplet loss with soft margin (Hermans et al., 'In Defense of the Triplet Loss')."""
    dist_mat = euclidean_dist(embedding, embedding)
    n = dist_mat.size(0)
    is_pos = targets.view(n, 1).expand(n, n).eq(targets.view(n, 1).expand(n, n).t()).float()
    is_neg = 1 - is_pos

    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

    y = torch.ones_like(dist_an)
    loss = F.soft_margin_loss(dist_an - dist_ap, y)
    if loss == float("Inf"):
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
    return loss


def cross_entropy_loss(pred_class_outputs, gt_classes):
    log_probs = F.log_softmax(pred_class_outputs, dim=1)
    loss = -log_probs.gather(1, gt_classes.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)
    return loss.sum() / non_zero_cnt


def loss_func(outputs, gt_labels, domain_labels, cfg):
    """L_total = lambda_id * L_id + lambda_tri * L_tri + lambda_dom * L_dom (Eq. 2)."""
    losses = {
        "ID_LOSS": cross_entropy_loss(outputs["cls_outputs"], gt_labels) * cfg.ID_LOSS_WEIGHT,
        "TRI_LOSS": triplet_loss(outputs["features"], gt_labels) * cfg.TRIPLET_LOSS_WEIGHT,
    }
    if cfg.DOMAIN_LOSS_WEIGHT > 0:
        losses["DOMAIN_LOSS"] = F.cross_entropy(outputs["domain_logits"], domain_labels) * cfg.DOMAIN_LOSS_WEIGHT
    losses["TOTAL_LOSS"] = sum(losses.values())
    return losses
