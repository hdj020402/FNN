"""Lightweight torch Dataset for FNN — no PyG dependency.

Loads features/targets from CSV, with optional ECFP fingerprint generation from SMILES.
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
    """Dataset of ``(features, target, weight)`` tuples.

    Features are read from CSV columns.  ECFP fingerprints are optionally
    generated from a SMILES column — no SDF / 3D coordinates needed.

    Args:
        data_file: CSV with feature, target, and optionally SMILES columns.
        feature_list: Column names for input features.
        target_list: Column names for prediction targets.
        weight_file: JSON file with per-sample weights (optional).
        mol_column: CSV column name containing SMILES (or InChI) strings.
            Required when ``ecfp`` is set; ignored otherwise.
        mol_format: ``'smiles'`` or ``'inchi'`` — how to parse the strings in
            ``mol_column``.
        ecfp: Dict with ``radius`` and ``nBits`` for ECFP fingerprints.
            ``None`` to disable.
    """

    def __init__(
        self,
        data_file: str,
        feature_list: list[str],
        target_list: list[str],
        weight_file: str | None = None,
        mol_column: str | None = None,
        mol_format: str = "smiles",
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

        # ECFP from SMILES / InChI
        if ecfp is not None:
            if mol_column is None:
                raise ValueError("mol_column is required when ECFP is enabled.")
            if mol_format not in ('smiles', 'inchi'):
                raise ValueError(f"Unknown mol_format '{mol_format}'. Use 'smiles' or 'inchi'.")
            radius = ecfp.get('radius', 2)
            nBits = ecfp.get('nBits', 1024)
            raw_series = database[mol_column]
            fps = []
            for raw in raw_series:
                if mol_format == 'smiles':
                    mol = Chem.MolFromSmiles(raw)
                else:
                    mol = Chem.MolFromInchi(raw)
                if mol is None:
                    raise ValueError(
                        f"RDKit could not parse {mol_format}: '{raw}'. "
                        f"Check the '{mol_column}' column in {data_file}."
                    )
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
    """Subset that exposes ``features``, ``targets``, ``weights`` tensor views."""

    def __init__(self, dataset: FeatureDataset, indices: list[int]) -> None:
        super().__init__(dataset, indices)
        self.features = dataset.features[indices]
        self.targets = dataset.targets[indices]
        self.weights = dataset.weights[indices]
