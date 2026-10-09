import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("reid.eval")

try:
    from reid.rank_cy import evaluate_cy
except ImportError:  # build with `make -C reid`, the Python version gives the same numbers, only slower
    evaluate_cy = None


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Market-1501 protocol. For each query, gallery images with the same identity and camera are discarded."""
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    indices = np.argsort(distmat, axis=1)

    all_cmc, all_AP, all_INP = [], [], []
    for q_idx in range(num_q):
        order = indices[q_idx]
        keep = ~((g_pids[order] == q_pids[q_idx]) & (g_camids[order] == q_camids[q_idx]))
        raw_cmc = (g_pids[order] == q_pids[q_idx]).astype(np.int32)[keep]
        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        max_pos_idx = np.max(np.where(raw_cmc == 1))
        all_INP.append(cmc[max_pos_idx] / (max_pos_idx + 1.0))
        cmc[cmc > 1] = 1
        all_cmc.append(np.pad(cmc[:max_rank], (0, max_rank - len(cmc[:max_rank])), mode="edge"))

        precision = raw_cmc.cumsum() / np.arange(1, len(raw_cmc) + 1)
        all_AP.append((precision * raw_cmc).sum() / raw_cmc.sum())

    assert len(all_cmc) > 0, "Error: all query identities do not appear in gallery"
    return np.asarray(all_cmc, dtype=np.float32).mean(0), all_AP, all_INP


@torch.no_grad()
def extract_features(model, loader):
    feats, pids, camids = [], [], []
    for data in loader:
        feats.append(model(data["x"].to("cuda")).float().cpu())
        pids.append(data["pid"])
        camids.append(data["camid"])
    return torch.cat(feats), torch.cat(pids).numpy(), torch.cat(camids).numpy()


@torch.no_grad()
def evaluate(model, query, gallery, epoch=0):
    """Cosine-distance retrieval of `query` against `gallery`, single forward pass, no re-ranking."""
    logger.info("Evaluating model")
    model.eval()
    q_feats, q_pids, q_camids = extract_features(model, query)
    g_feats, g_pids, g_camids = extract_features(model, gallery)

    distmat = (1 - F.normalize(q_feats, dim=1) @ F.normalize(g_feats, dim=1).t()).numpy()
    if evaluate_cy is not None:
        cmc, all_AP, _ = evaluate_cy(distmat, q_pids, g_pids, q_camids, g_camids, 50, False)
    else:
        cmc, all_AP, _ = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids)

    rank1, rank5, rank10, mAP = cmc[0], cmc[4], cmc[9], np.mean(all_AP)
    print(f"Validation Results - Epoch: {epoch}")
    print(f"mAP: {mAP:.2%}")
    print(f"CMC curve, Rank-1 : {rank1:.2%}")
    print(f"CMC curve, Rank-5 : {rank5:.2%}")
    print(f"CMC curve, Rank-10: {rank10:.2%}")
    return rank1, rank5, rank10, mAP
