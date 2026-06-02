"""Lightweight torch Dataset for FNN — no PyG dependency.

Supports loading features/targets from CSV, with optional ECFP fingerprint generation from SDF.
"""
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem


class FeatureDataset(Dataset):
    """Dataset that loads features and targets from CSV, with optional ECFP generation.

    Each sample is a ``(features, target, weight)`` tuple of tensors.

    Args:
        data_file: CSV file with feature and target columns.
        feature_list: Column names to use as input features.
        target_list: Column names to use as prediction targets.
        sdf_file: SDF file for ECFP generation (required if ``ecfp`` is not None).
        weight_file: JSON file with per-sample weights (optional).
        ecfp: Dict with ``radius`` and ``nBits`` for ECFP fingerprints.  None to disable.
    """

    def __init__(
        self,
        data_file: str,
        feature_list: list[str],
        target_list: list[str],
        sdf_file: str | None = None,
        weight_file: str | None = None,
        ecfp: dict | None = None,
    ):
        self.data_file = data_file
        self.feature_list = feature_list
        self.target_list = target_list
        self.num_features = len(feature_list)
        self.num_targets = len(target_list)

        database = pd.read_csv(data_file)

        # Targets
        self.targets = torch.tensor(
            np.array(database.loc[:, target_list]),
            dtype=torch.float,
        ).reshape(-1, len(target_list))

        # Features (from CSV columns)
        if feature_list:
            self.features = torch.tensor(
                np.array(database.loc[:, feature_list]),
                dtype=torch.float,
            ).reshape(-1, len(feature_list))
        else:
            self.features = torch.empty(len(self.targets), 0)

        # ECFP fingerprints
        if ecfp is not None:
            suppl = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False)
            radius = ecfp.get('radius', 2)
            nBits = ecfp.get('nBits', 1024)
            fps = []
            for mol in suppl:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useChirality=True)
                fps.append(torch.tensor(fp, dtype=torch.float))
            ecfp_tensor = torch.stack(fps)  # [N, nBits]
            self.features = torch.cat([self.features, ecfp_tensor], dim=1)
            self.num_features += nBits

        # Weights
        if weight_file is not None:
            with open(weight_file) as wf:
                raw_weights = json.load(wf)
            self.weights = torch.tensor(raw_weights, dtype=torch.float)
            if self.weights.dim() == 1:
                self.weights = self.weights.reshape(-1, 1)
        else:
            self.weights = torch.ones(len(self.targets), 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx], self.weights[idx]

    def __len__(self) -> int:
        return len(self.features)


class FeatureSubset(Subset):
    """Subset that retains ``features``, ``targets``, ``weights`` tensor views for mean/std computation."""

    def __init__(self, dataset: FeatureDataset, indices: list[int]) -> None:
        super().__init__(dataset, indices)
        self.features = dataset.features[indices]
        self.targets = dataset.targets[indices]
        self.weights = dataset.weights[indices]
