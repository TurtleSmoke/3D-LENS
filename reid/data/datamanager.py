from pathlib import Path

import torch
import torch.utils.data

from reid.data.datasets import build_datasets
from reid.data.sampler import BalancedDomainSampler, NaiveIdentitySampler
from reid.data.transforms import build_transforms


class DataManager:
    def __init__(self, cfg):
        self._reid_dataset = build_datasets(
            cfg.DATA.DATASET,
            [Path(cfg.DATA.ROOT_DIR) / d for d in cfg.DATA.DATASET_DIR],
            build_transforms(cfg.TRAIN.IMG_SIZE, is_train=True),
            build_transforms(cfg.TRAIN.IMG_SIZE, is_train=False),
            train=cfg.DATA.TRAIN,
            query=cfg.DATA.QUERY,
            gallery=cfg.DATA.GALLERY,
            domain_filter=cfg.DATA.DOMAIN_FILTER,
        )

        train_data = self._reid_dataset.train.data
        if cfg.TRAIN.SAMPLER == "balanced":
            sampler = BalancedDomainSampler(
                train_data,
                cfg.TRAIN.BATCH_SIZE,
                cfg.TRAIN.NUM_INSTANCES,
                seed=cfg.GLOBAL.SEED,
                real_ratio=cfg.TRAIN.REAL_RATIO,
                warmup_epochs=cfg.TRAIN.CURRICULUM_WARMUP_EPOCHS,
            )
        elif cfg.TRAIN.SAMPLER == "naive":
            sampler = NaiveIdentitySampler(
                train_data, cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.NUM_INSTANCES, seed=cfg.GLOBAL.SEED
            )
        else:
            raise ValueError(f"Unknown sampler '{cfg.TRAIN.SAMPLER}', expected 'balanced' or 'naive'")

        self.train_sampler = sampler
        self.train_loader = torch.utils.data.DataLoader(
            self._reid_dataset.train,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.GLOBAL.NUM_WORKERS,
            sampler=sampler,
        )

        self.test_loader = {
            name: tuple(
                torch.utils.data.DataLoader(
                    dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=cfg.GLOBAL.NUM_WORKERS
                )
                for dataset in (query_set, gallery_set)
            )
            for name, (query_set, gallery_set) in self._reid_dataset.test_scenarios.items()
        }

    def summary(self):
        self._reid_dataset.summary()
        print(f"Train loader: {len(self.train_loader)} batches")
        for name, (query, gallery) in self.test_loader.items():
            print(f"Test loader '{name}': {len(query)} query batches, {len(gallery)} gallery batches")
