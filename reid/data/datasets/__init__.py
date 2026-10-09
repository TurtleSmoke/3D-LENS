from .ReIDDataset import AGReIDDataset, AGReIDv2Dataset, MOODataset

_DATASETS = {
    "AG-ReID": AGReIDDataset,
    "AG-ReID.v2": AGReIDv2Dataset,
    "MOO": MOODataset,
}


def build_datasets(name, dataset_dir, train_transform, test_transform, **kwargs):
    if name not in _DATASETS:
        raise KeyError(f"Invalid dataset name '{name}', expected one of {list(_DATASETS)}")
    return _DATASETS[name](dataset_dir, train_transform, test_transform, **kwargs)
