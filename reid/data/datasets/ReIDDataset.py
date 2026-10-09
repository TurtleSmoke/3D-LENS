import json
import re
from collections import Counter
from pathlib import Path

from reid.data.datasets.ImageDataset import ImageDataset

AG_PATTERN = re.compile(r"P(\d+)T(\d+)A(\d+)C(\d+)F\d+")
MOO_PATTERN = re.compile(r"cow(\d+)_(\d+)")


def _read_lists(dirs, list_name, img_dir=""):
    """
    Read the image list `<dir>/<list_name>` of [real] or [real, synthetic] as sorted (path, domain) pairs.
    The listed paths are relative to `<dir>/<img_dir>`.
    """
    return sorted(
        (d / img_dir / name, domain)
        for domain, d in enumerate(dirs)
        for name in (d / list_name).read_text().splitlines()
    )


def _build(imgs, pid_fn, camid_fn, domain_fn, domain_filter=None, train=False):
    """
    Turn a list of (path, domain) pairs into (path, domain, pid, camid) tuples.

    `domain` is the real (0) / synthetic (1) label used by the domain loss and the sampler.
    domain_fn gives the viewpoint-domain of an image (e.g. aerial / ground) used by DOMAIN_FILTER.
    """
    imgs = [(str(img), domain) for img, domain in imgs]
    assert len(imgs) > 0, "No images found"

    if domain_filter is not None:
        imgs = [(img, domain) for img, domain in imgs if domain_fn(img) in domain_filter]

    data = [(img, domain, pid_fn(img), camid_fn(img)) for img, domain in imgs]

    if train:
        counter = Counter(pid for _, _, pid, _ in data)
        data = [d for d in data if counter[d[2]] >= 4]  # The P x K sampler needs at least K=4 images per identity
        # CE targets index the ID classifier, so pids must be contiguous 0..N-1 after the filter above
        relabel = {pid: i for i, pid in enumerate(sorted({d[2] for d in data}))}
        data = [(img, domain, relabel[pid], camid) for img, domain, pid, camid in data]

    synthetic = sum(d[1] for d in data)
    print(f"Loaded {len(data)} images: {len(data) - synthetic} real and {synthetic} synthetic")

    return data


def _ag_pid(img):
    return int("".join(AG_PATTERN.findall(img)[0][:3]))


def _ag_camid(img):
    return int(AG_PATTERN.findall(img)[0][3])


class ImageReIDDataset:
    """
    dataset_dir is [real] or [real, synthetic (3D-LENS)].
    Train images come from both directories, query and gallery only from the real one.
    """

    def __init__(self, dataset_dir, train_transform, test_transform, train, query, gallery, domain_filter=None):
        assert len(dataset_dir) in (1, 2), f"Expected [real] or [real, synthetic], got {dataset_dir}"
        self._dataset_dir = [Path(d) for d in dataset_dir]
        self._train_name = train
        self._query_name = query
        self._gallery_name = gallery
        self._domain_filter = domain_filter

        train_data, query_data, gallery_data = self._load_datasets()

        if len(query_data) != len(gallery_data):
            raise ValueError(f"Query and gallery must have the same length: {len(query_data)} != {len(gallery_data)}")

        self.train = ImageDataset(train_data, train_transform, name="train")
        self.query = [ImageDataset(d, test_transform, name=n) for d, n in zip(query_data, query, strict=True)]
        self.gallery = [ImageDataset(d, test_transform, name=n) for d, n in zip(gallery_data, gallery, strict=True)]

    @property
    def test_scenarios(self):
        """Each scenario is a (query, gallery) pair of datasets."""
        return {f"{q.name}_{g.name}": (q, g) for q, g in zip(self.query, self.gallery, strict=True)}

    def _load_datasets(self):
        raise NotImplementedError("Must be implemented in subclass")

    def summary(self):
        print("===== Image ReID Dataset Summary =====")
        print(f"Dataset directory: {self._dataset_dir}")
        print(f"Train dataset: {len(self.train)} images ({len(set(self.train.pids))} unique IDs)")
        for name, (query, gallery) in self.test_scenarios.items():
            print(f"Test scenario '{name}': {len(query)} query images and {len(gallery)} gallery images")
            print(f"  Query IDs: {len(set(query.pids))}, Gallery IDs: {len(set(gallery.pids))}")
        print("======================================")


class AGReIDDataset(ImageReIDDataset):
    """AG-ReID. Camera codes are 0 = aerial (UAV), 3 = ground (CCTV)."""

    SCENARIOS = {
        ("aerial", "cctv"): ("query_all_c0", "bounding_box_test_all_c3"),
        ("cctv", "aerial"): ("query_all_c3", "bounding_box_test_all_c0"),
    }

    def _load_datasets(self):
        real = self._dataset_dir[0]
        build = dict(pid_fn=_ag_pid, camid_fn=_ag_camid, domain_fn=_ag_camid)

        train = _build(
            _read_lists(self._dataset_dir, self._train_name), **build, domain_filter=self._domain_filter, train=True
        )

        def load_split(split):
            imgs = sorted(real.glob(f"{split}/*.jpg")) + sorted(real.glob(f"{split}/*.png"))
            return _build([(img, 0) for img in imgs], **build)

        scenarios = [self.SCENARIOS[(q, g)] for q, g in zip(self._query_name, self._gallery_name, strict=True)]
        return train, [load_split(q) for q, _ in scenarios], [load_split(g) for _, g in scenarios]


class AGReIDv2Dataset(ImageReIDDataset):
    """AG-ReID.v2. Camera codes are 0 = aerial (UAV), 2 = wearable, 3 = CCTV."""

    SCENARIOS = {
        ("aerial", "cctv"): "exp1_aerial_to_cctv.txt",
        ("aerial", "wearable"): "exp2_aerial_to_wearable.txt",
        ("cctv", "aerial"): "exp4_cctv_to_aerial.txt",
        ("wearable", "aerial"): "exp5_wearable_to_aerial.txt",
    }

    def _load_datasets(self):
        real = self._dataset_dir[0]
        build = dict(pid_fn=_ag_pid, camid_fn=_ag_camid, domain_fn=_ag_camid)

        # The lists sit in <dataset_dir>, their image paths are relative to <dataset_dir>/AG-ReID.v2
        img_dir = "AG-ReID.v2"

        train = _build(
            _read_lists(self._dataset_dir, self._train_name, img_dir),
            **build,
            domain_filter=self._domain_filter,
            train=True,
        )

        def load_split(scenario, split):
            # A scenario list holds both its query/ and gallery/ images
            names = (real / scenario).read_text().splitlines()
            return _build(sorted((real / img_dir / n, 0) for n in names if n.startswith(f"{split}/")), **build)

        scenarios = [self.SCENARIOS[(q, g)] for q, g in zip(self._query_name, self._gallery_name, strict=True)]
        return train, [load_split(s, "query") for s in scenarios], [load_split(s, "gallery") for s in scenarios]


class MOODataset(ImageReIDDataset):
    """MOO (cattle). Viewpoint domain is 0 = side, 1 = top (frame index >= 64)."""

    def _load_datasets(self):
        real = self._dataset_dir[0]

        def load(imgs, camid, domain_filter=None, train=False):
            return _build(
                imgs,
                pid_fn=lambda img: int(MOO_PATTERN.findall(Path(img).name)[0][0]),
                camid_fn=lambda img: camid,
                domain_fn=lambda img: int(int(MOO_PATTERN.findall(Path(img).name)[0][1]) >= 64),
                domain_filter=domain_filter,
                train=train,
            )

        def read_json(split):
            entries = json.load(open(real / "output_screenshots_split" / f"{split}.json"))
            return sorted((str(real / e["filename"]), 0) for e in entries if (real / e["filename"]).is_file())

        train_ids = {int(e["ID"]) for e in json.load(open(real / "output_screenshots_split" / "train_all.json"))}
        train_imgs = [
            (img, domain)
            for img, domain in _read_lists(self._dataset_dir, self._train_name)
            if int(MOO_PATTERN.findall(img.name)[0][0]) in train_ids
        ]
        train = load(train_imgs, 0, self._domain_filter, train=True)
        # Query and gallery use different camids so that the same-camera rule of Market-1501 discards no gallery image
        query = [load(read_json(f"test_{s}_query"), 0) for s in self._query_name]
        gallery = [load(read_json(f"test_{s}_gallery"), 1) for s in self._gallery_name]
        return train, query, gallery
