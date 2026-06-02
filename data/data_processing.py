"""Data processing pipeline — dataset creation, splitting, normalization, loaders.

Uses standard ``torch.utils.data`` (no PyG dependency).
"""
import torch
import numpy as np
from torch.utils.data import random_split, DataLoader

from data.dataset import FeatureDataset, FeatureSubset
from configs.schema import ModelParams


class DataProcessing:
    """Handles dataset loading, splitting, normalization, and DataLoader creation.

    Args:
        param: ModelParams configuration.
    """

    def __init__(self, param: ModelParams) -> None:
        self.param = param
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = self._gen_dataset()
        self.train_dataset, self.val_dataset, self.test_dataset, self.pred_dataset = self._split_dataset()
        self.norm_dict = self._get_mean_std()
        self._normalization()
        self.train_loader, self.val_loader, self.test_loader, self.pred_loader = self._gen_loaders()

    def _gen_dataset(self) -> FeatureDataset:
        """Create the FeatureDataset from config."""
        p = self.param
        rdkit_cfg = self._build_rdkit_config()

        dataset = FeatureDataset(
            data_file=p.data_file,
            feature_list=list(p.feature_list),
            target_list=list(p.target_list),
            weight_file=p.weight_file,
            rdkit=rdkit_cfg,
        )
        dataset = self._target_transform(dataset)
        return dataset

    def _build_rdkit_config(self) -> dict | None:
        """Build the rdkit config dict from the nested dataclass structure.

        Returns None when rdkit is disabled (no molecular features needed).
        """
        rdkit = self.param.default_feature.rdkit
        if not rdkit.enabled:
            return None
        return {
            'enabled': True,
            'mol_column': rdkit.mol_column,
            'mol_format': rdkit.mol_format,
            'ecfp': {
                'enabled': rdkit.ecfp.enabled,
                'radius': rdkit.ecfp.radius,
                'nBits': rdkit.ecfp.nBits,
            },
            'descriptors': {
                'enabled': rdkit.descriptors.enabled,
                'include': rdkit.descriptors.include,
            },
        }

    def _target_transform(self, dataset: FeatureDataset) -> FeatureDataset:
        """Apply target transformation in-place."""
        transform = self.param.target_transform
        if transform == 'LN':
            dataset.targets = torch.log(dataset.targets)
        elif transform == 'LG':
            dataset.targets = torch.log10(dataset.targets)
        elif transform == 'E^-x':
            dataset.targets = torch.exp(-dataset.targets)
        elif not transform:
            pass
        return dataset

    def _split_dataset(self) -> tuple:
        """Split dataset into train/val/test sets."""
        pred_dataset = self.dataset

        if self.param.split_method == 'random':
            train_size = int(self.param.train_size * len(self.dataset))
            val_size = int(self.param.val_size * len(self.dataset))
            test_size = len(self.dataset) - train_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.param.seed),
            )
        elif self.param.split_method == 'manual':
            indices = np.load(self.param.split_file, allow_pickle=True)
            train_dataset = FeatureSubset(self.dataset, indices[0].tolist())
            val_dataset = FeatureSubset(self.dataset, indices[1].tolist())
            test_dataset = FeatureSubset(self.dataset, indices[2].tolist())
        else:
            raise NotImplementedError("Split method not implemented.")

        return train_dataset, val_dataset, test_dataset, pred_dataset

    def _gen_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders for train/val/test/pred."""
        dl_kwargs = dict(
            batch_size=self.param.batch_size,
            num_workers=self.param.num_workers,
            pin_memory=True,
        )
        train_loader = DataLoader(self.train_dataset, shuffle=True, **dl_kwargs)
        val_loader = DataLoader(self.val_dataset, shuffle=False, **dl_kwargs)
        test_loader = DataLoader(self.test_dataset, shuffle=False, **dl_kwargs)
        pred_loader = DataLoader(self.pred_dataset, shuffle=False, **dl_kwargs)
        return train_loader, val_loader, test_loader, pred_loader

    def _normalization(self) -> None:
        """Normalize features and targets using mean/std from training set."""
        mean_f, std_f = self.norm_dict.get('feature', (None, None))
        mean_y, std_y = self.norm_dict['y']
        if hasattr(self.dataset, 'features'):
            if mean_f is not None:
                self.dataset.features = (self.dataset.features - mean_f.squeeze(0)) / std_f.squeeze(0)
            self.dataset.targets = (self.dataset.targets - mean_y.squeeze(0)) / std_y.squeeze(0)

    def _get_mean_std(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Compute mean and std from training set (or load from pretrained model)."""
        if self.param.mode == 'prediction':
            pretrained_model = self.param.pretrained_model
            state_dict: dict = torch.load(pretrained_model, map_location=torch.device('cpu'), weights_only=False)
            return state_dict['norm']
        else:
            indices = self.train_dataset.indices
            train_feat = self.dataset.features[indices]
            train_y = self.dataset.targets[indices]

            norm_dict: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            if self.param.feature_list:
                mean = train_feat.mean(dim=0, keepdim=True)
                std = train_feat.std(dim=0, keepdim=True).clamp(min=1e-8)
                norm_dict['feature'] = (mean, std)
            y_mean = train_y.mean(dim=0, keepdim=True)
            y_std = train_y.std(dim=0, keepdim=True).clamp(min=1e-8)
            norm_dict['y'] = (y_mean, y_std)
            return norm_dict

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def mean(self) -> torch.Tensor:
        """Mean of y (for backward compat)."""
        return self.norm_dict['y'][0]

    @property
    def std(self) -> torch.Tensor:
        """Std of y (for backward compat)."""
        return self.norm_dict['y'][1]
