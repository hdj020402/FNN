"""Lightweight torch Dataset for FNN — no PyG dependency.

Loads features/targets from CSV.  When ``rdkit`` is enabled, ECFP fingerprints
and/or RDKit 2-D descriptors are computed from a molecular string column
(SMILES or InChI).
"""
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Descriptors, AllChem

from configs.rdkit_descriptors import RECOMMENDED, ALL_2D


def _resolve_descriptor_names(spec: list[str] | str | None) -> list[str]:
    """Resolve a descriptor include spec to a flat list of RDKit descriptor names.

    * ``None`` or absent  → ``RECOMMENDED``
    * ``'all_2d'``        → every descriptor in ``ALL_2D``
    * ``['MolLogP', ...]``→ used as-is (caller is responsible for validity)
    """
    if spec is None:
        return list(RECOMMENDED)
    if isinstance(spec, str) and spec == 'all_2d':
        names: list[str] = []
        for items in ALL_2D.values():
            for name, _ in items:
                names.append(name)
        return names
    if isinstance(spec, list):
        return spec
    raise ValueError(
        f"Invalid descriptors.include value: {spec!r}. "
        f"Use null, 'all_2d', or a list of descriptor names."
    )


def _compute_descriptors(mols: list, desc_names: list[str]) -> torch.Tensor:
    """Compute a descriptor matrix [N, D] from a list of RDKit Mol objects."""
    rows: list[list[float]] = []
    for mol in mols:
        row = []
        for name in desc_names:
            func = getattr(Descriptors, name, None)
            if func is None:
                raise ValueError(
                    f"Unknown RDKit descriptor: '{name}'. "
                    f"See configs/rdkit_descriptors.py for the full list."
                )
            row.append(float(func(mol)))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float)


class FeatureDataset(Dataset):
    """Dataset of ``(features, target, weight)`` tuples.

    Features are read from CSV columns.  RDKit molecular features (ECFP +
    descriptors) can be enabled via the ``rdkit`` dict.

    Args:
        data_file: CSV with feature, target, and optionally molecular columns.
        feature_list: Column names for input features.
        target_list: Column names for prediction targets.
        weight_file: JSON file with per-sample weights (optional).
        rdkit: Configuration dict for RDKit feature generation (see below).
            ``None`` to disable all RDKit features.

    The ``rdkit`` dict has the structure::

        {
            "enabled": True,
            "mol_column": "SMILES",
            "mol_format": "smiles",      # or "inchi"
            "ecfp": {
                "enabled": True,
                "radius": 2,
                "nBits": 1024,
            },
            "descriptors": {
                "enabled": True,
                "include": None,          # None=recommended, 'all_2d', or [...]
            },
        }
    """

    def __init__(
        self,
        data_file: str,
        feature_list: list[str],
        target_list: list[str],
        weight_file: str | None = None,
        rdkit: dict | None = None,
    ):
        self.data_file = data_file
        self.feature_list = feature_list
        self.target_list = target_list
        self.num_features = len(feature_list)
        self.num_targets = len(target_list)

        database = pd.read_csv(data_file)

        # ── Targets ────────────────────────────────────────────────────────
        self.targets = torch.tensor(
            np.array(database.loc[:, target_list]),
            dtype=torch.float,
        ).reshape(-1, len(target_list))

        # ── Features (CSV columns) ─────────────────────────────────────────
        if feature_list:
            self.features = torch.tensor(
                np.array(database.loc[:, feature_list]),
                dtype=torch.float,
            ).reshape(-1, len(feature_list))
        else:
            self.features = torch.empty(len(self.targets), 0)

        # ── RDKit molecular features ──────────────────────────────────────
        if rdkit is not None and rdkit.get('enabled', False):
            self.features = self._add_rdkit_features(database, rdkit)

        # ── Weights ────────────────────────────────────────────────────────
        if weight_file is not None:
            with open(weight_file) as wf:
                raw_weights = json.load(wf)
            self.weights = torch.tensor(raw_weights, dtype=torch.float)
            if self.weights.dim() == 1:
                self.weights = self.weights.reshape(-1, 1)
        else:
            self.weights = torch.ones(len(self.targets), 1)

    # ── RDKit feature helpers ──────────────────────────────────────────────

    def _parse_mols(self, database: pd.DataFrame, rdkit: dict) -> list:
        """Parse molecular strings into RDKit Mol objects."""
        mol_column = rdkit.get('mol_column')
        if mol_column is None:
            raise ValueError("rdkit.mol_column is required when rdkit.enabled is True.")
        mol_format = rdkit.get('mol_format', 'smiles')
        if mol_format not in ('smiles', 'inchi'):
            raise ValueError(f"Unknown rdkit.mol_format '{mol_format}'. Use 'smiles' or 'inchi'.")

        raw_series = database[mol_column]
        mols = []
        for raw in raw_series:
            if mol_format == 'smiles':
                mol = Chem.MolFromSmiles(raw)
            else:
                mol = Chem.MolFromInchi(raw)
            if mol is None:
                raise ValueError(
                    f"RDKit could not parse {mol_format}: '{raw}'. "
                    f"Check the '{mol_column}' column in {self.data_file}."
                )
            mols.append(mol)
        return mols

    def _add_rdkit_features(self, database: pd.DataFrame, rdkit: dict) -> torch.Tensor:
        """Compute ECFP and/or descriptor features and concatenate them."""
        mols = self._parse_mols(database, rdkit)
        feats = [self.features]

        # ECFP
        ecfp_cfg = rdkit.get('ecfp', {})
        if ecfp_cfg.get('enabled', False):
            radius = ecfp_cfg.get('radius', 2)
            nBits = ecfp_cfg.get('nBits', 1024)
            fps = []
            for mol in mols:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useChirality=True)
                fps.append(torch.tensor(fp, dtype=torch.float))
            ecfp_tensor = torch.stack(fps)
            feats.append(ecfp_tensor)
            self.num_features += nBits

        # Descriptors
        desc_cfg = rdkit.get('descriptors', {})
        if desc_cfg.get('enabled', False):
            desc_names = _resolve_descriptor_names(desc_cfg.get('include'))
            desc_tensor = _compute_descriptors(mols, desc_names)
            feats.append(desc_tensor)
            self.num_features += desc_tensor.size(1)

        return torch.cat(feats, dim=1)

    # ── Dataset protocol ───────────────────────────────────────────────────

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
