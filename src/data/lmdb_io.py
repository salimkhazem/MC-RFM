"""LMDB I/O for cached frozen features."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


def _encode_array(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _decode_array(raw: bytes) -> np.ndarray:
    return np.load(io.BytesIO(raw), allow_pickle=False)


@dataclass
class LMDBMeta:
    num_samples: int
    feature_dim: int
    dataset: str
    split: str
    backbone: str


class FeatureLMDBWriter:
    def __init__(self, path: str | Path, map_size_gb: int = 16):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(
            str(self.path),
            map_size=int(map_size_gb * (1024**3)),
            subdir=False,
            lock=False,
            readonly=False,
            readahead=False,
            meminit=False,
            sync=False,
            metasync=False,
        )

    def put(self, index: int, feature: np.ndarray, label: int) -> None:
        key_f = f"f:{index}".encode("utf-8")
        key_l = f"y:{index}".encode("utf-8")
        with self.env.begin(write=True) as txn:
            txn.put(key_f, _encode_array(feature.astype(np.float32)))
            txn.put(key_l, np.asarray([label], dtype=np.int64).tobytes())

    def put_many(self, entries: list[tuple[int, np.ndarray, int]]) -> None:
        with self.env.begin(write=True) as txn:
            for index, feature, label in entries:
                key_f = f"f:{index}".encode("utf-8")
                key_l = f"y:{index}".encode("utf-8")
                txn.put(key_f, _encode_array(feature.astype(np.float32)))
                txn.put(key_l, np.asarray([label], dtype=np.int64).tobytes())

    def put_meta(self, meta: LMDBMeta) -> None:
        with self.env.begin(write=True) as txn:
            txn.put(b"meta", json.dumps(meta.__dict__).encode("utf-8"))

    def close(self) -> None:
        self.env.close()


class FeatureLMDBReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"LMDB not found: {self.path}")
        self.env = lmdb.open(
            str(self.path),
            subdir=False,
            lock=False,
            readonly=True,
            readahead=True,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            raw = txn.get(b"meta")
            if raw is None:
                raise RuntimeError(f"Missing meta in LMDB: {self.path}")
            payload = json.loads(raw.decode("utf-8"))
            self.meta = LMDBMeta(**payload)

    def get(self, index: int) -> tuple[np.ndarray, int]:
        key_f = f"f:{index}".encode("utf-8")
        key_l = f"y:{index}".encode("utf-8")
        with self.env.begin(write=False) as txn:
            f_raw = txn.get(key_f)
            y_raw = txn.get(key_l)
        if f_raw is None or y_raw is None:
            raise KeyError(f"Sample {index} not found in {self.path}")
        feat = _decode_array(f_raw).astype(np.float32)
        label = int(np.frombuffer(y_raw, dtype=np.int64)[0])
        return feat, label

    def close(self) -> None:
        self.env.close()


class LMDBFeatureDataset(Dataset):
    def __init__(self, path: str | Path, indices: list[int] | None = None):
        self.path = Path(path)
        reader = FeatureLMDBReader(path)
        self.meta = reader.meta
        reader.close()
        self.indices = indices if indices is not None else list(range(self.meta.num_samples))
        self._reader: FeatureLMDBReader | None = None

    def _get_reader(self) -> FeatureLMDBReader:
        if self._reader is None:
            self._reader = FeatureLMDBReader(self.path)
        return self._reader

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        feat, label = self._get_reader().get(idx)
        return torch.from_numpy(feat).float(), int(label), int(idx)

    @property
    def feature_dim(self) -> int:
        return int(self.meta.feature_dim)

    @property
    def num_samples(self) -> int:
        return int(self.meta.num_samples)

    @property
    def num_classes_hint(self) -> int:
        # only a hint, not guaranteed exact without full scan
        return -1
