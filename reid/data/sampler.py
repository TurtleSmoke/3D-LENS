import copy
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data.sampler import Sampler


class NaiveIdentitySampler(Sampler):
    """
    Randomly sample P identities, then K instances per identity, therefore batch size is P*K.
    Used for the ViT baseline (real images only).

    Args:
    - data_source (list): list of (path, domain, pid, camid).
    - mini_batch_size (int): number of examples in a batch.
    - num_instances (int): number of instances per identity in a batch (K).
    """

    def __init__(self, data_source, mini_batch_size, num_instances, seed=42):
        self.num_instances = num_instances
        self.num_pids_per_batch = mini_batch_size // num_instances
        self.batch_size = mini_batch_size
        self._seed = int(seed)

        self.pid_index = defaultdict(list)
        for index, (_, _, pid, _) in enumerate(data_source):
            self.pid_index[pid].append(index)
        self.pids = sorted(self.pid_index)

        self.length = 0
        for pid in self.pids:
            num = max(len(self.pid_index[pid]), self.num_instances)
            self.length += num - num % self.num_instances

    def __iter__(self):
        np.random.seed(self._seed)
        while True:
            avl_pids = copy.deepcopy(self.pids)
            batch_idxs_dict = {}
            batch_indices = []
            while len(avl_pids) >= self.num_pids_per_batch:
                selected_pids = np.random.choice(avl_pids, self.num_pids_per_batch, replace=False).tolist()
                for pid in selected_pids:
                    if pid not in batch_idxs_dict:
                        batch_idxs_dict[pid] = self._sample_identity(pid)

                    avl_idxs = batch_idxs_dict[pid]
                    for _ in range(self.num_instances):
                        batch_indices.append(avl_idxs.pop(0))

                    if len(avl_idxs) < self.num_instances:
                        avl_pids.remove(pid)

                if len(batch_indices) == self.batch_size:
                    yield from batch_indices
                    batch_indices = []

    def _sample_identity(self, pid):
        idxs = copy.deepcopy(self.pid_index[pid])
        if len(idxs) < self.num_instances:
            idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
        np.random.shuffle(idxs)
        return idxs

    def __len__(self):
        return self.length


class BalancedDomainSampler(NaiveIdentitySampler):
    """
    Balanced sampling (Sec. 3.3). Each identity contributes K_real = rho * K real images and
    K_syn = K - K_real synthetic views, regardless of how many synthetic views exist.
    One epoch is one pass over the real images.

    Curriculum scheduler (Sec. 3.3). A synthetic view drawn at epoch e is rejected with probability
        P(dtheta, e) = max(0, 1 - e / E_warmup) * |dtheta| / |dtheta_max|
    The elevation shift of a view is read from its `_XX` suffix. Views _00 ... _NN are rendered with
    |dtheta| linearly spaced in [0, dtheta_max], so |dtheta| / |dtheta_max| = XX / NN.
    E_warmup = 0 disables the curriculum.
    """

    def __init__(self, data_source, mini_batch_size, num_instances, seed=42, real_ratio=0.5, warmup_epochs=20):
        super().__init__(data_source, mini_batch_size, num_instances, seed)
        self.warmup_epochs = warmup_epochs
        self.epoch = 1

        # pid -> {0: [real indices], 1: [synthetic indices]}
        self.pid_domain_index = defaultdict(lambda: {0: [], 1: []})
        view_ids = np.zeros(len(data_source))
        for index, (path, domain, pid, _) in enumerate(data_source):
            self.pid_domain_index[pid][domain].append(index)
            if domain == 1:
                view_ids[index] = int(re.search(r"_(\d+)$", Path(path).stem).group(1))
        # Normalized elevation shift |dtheta| / |dtheta_max| of each sample (0 for real images)
        self.difficulty = view_ids / max(view_ids.max(), 1)

        self.num_real = int(num_instances * real_ratio)
        self.num_synthetic = num_instances - self.num_real

        total_real = sum(len(self.pid_domain_index[pid][0]) for pid in self.pids)
        self.length = int(total_real / real_ratio) if real_ratio > 0 else total_real

        print(
            f"BalancedDomainSampler: {self.num_real}/{num_instances} real and {self.num_synthetic}/{num_instances} "
            f"synthetic per ID, epoch length={self.length}, curriculum warmup={warmup_epochs} epochs"
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def rejection_probability(self, idxs):
        if self.warmup_epochs <= 0:
            return np.zeros(len(idxs))
        return max(0.0, 1 - self.epoch / self.warmup_epochs) * self.difficulty[idxs]

    def _sample_synthetic(self, synth_idxs, k):
        picked = []
        for _ in range(100):
            candidates = np.random.choice(synth_idxs, size=k, replace=len(synth_idxs) < k)
            accepted = candidates[np.random.rand(k) >= self.rejection_probability(candidates)]
            picked.extend(accepted[: k - len(picked)].tolist())
            if len(picked) == k:
                return picked
        # Only reachable if an identity has no near-source view (_00)
        return picked + np.random.choice(synth_idxs, size=k - len(picked), replace=True).tolist()

    def _sample_identity(self, pid):
        real_idxs = self.pid_domain_index[pid][0]
        synth_idxs = self.pid_domain_index[pid][1]

        idxs = []
        if len(real_idxs) > 0 and self.num_real > 0:
            idxs += np.random.choice(real_idxs, size=self.num_real, replace=len(real_idxs) < self.num_real).tolist()
        if len(synth_idxs) > 0 and self.num_synthetic > 0:
            idxs += self._sample_synthetic(synth_idxs, self.num_synthetic)

        if len(idxs) < self.num_instances:
            idxs += np.random.choice(self.pid_index[pid], size=self.num_instances - len(idxs), replace=True).tolist()

        np.random.shuffle(idxs)
        return idxs
