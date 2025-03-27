import torch
import os, shutil, json
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset
from typing import List, Optional, Callable, Dict

class CustomDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        data_file: str,
        default_feature: Dict,
        feature_list: List[str],
        target_list: List[str],
        sdf_file: str=None,
        weight_file: str=None,
        transform: Optional[Callable]=None,
        pre_transform: Optional[Callable]=None,
        pre_filter: Optional[Callable]=None,
        reprocess: bool=False,
        ):
        self.root = root
        self.data_file = data_file
        self.sdf_file = sdf_file
        self.weight_file = weight_file
        self.default_feature = default_feature
        self.feature_list = feature_list
        self.target_list = target_list
        self.reprocess = reprocess
        if self.reprocess:
            self._reprocess()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def _reprocess(self):
        if os.path.exists(os.path.join(self.root, 'processed/')):
            shutil.rmtree(os.path.join(self.root, 'processed/'))

    def process(self):
        data_list = []
        database = pd.read_csv(self.data_file)
        target = torch.tensor(
            np.array(database.loc[:, self.target_list]),
            dtype = torch.float
            ).reshape(-1, len(self.target_list)).unsqueeze(1)
        if self.feature_list:
            feature = torch.tensor(
                np.array(database.loc[:, self.feature_list]),
                dtype = torch.float
                ).reshape(-1, len(self.feature_list)).unsqueeze(1)
        else:
            feature = torch.empty(len(target), 1, 0)

        if self.weight_file is None:
            weights = [1] * len(database)
        else:
            with open(self.weight_file) as wf:
                weights = json.load(wf)

        if self.default_feature['ECFP']['enabled'] is True:
            radius = self.default_feature['ECFP']['radius']
            nBits = self.default_feature['ECFP']['nBits']
            suppl = Chem.SDMolSupplier(
                self.sdf_file,
                removeHs=False,
                sanitize=False,
                )
            for i, (_feature, mol, _target, weight) in enumerate(zip(feature, suppl, target, weights)):
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useChirality=True)
                fp = torch.tensor(fp, dtype = torch.float).reshape(1, nBits)
                _feature = torch.cat([_feature, fp], axis=1)
                data = Data(feature=_feature, y=_target, weight=weight)
                data_list.append(data)
        else:
            for i, (_feature, _target, weight) in enumerate(zip(feature, target, weights)):
                data = Data(feature=_feature, y=_target, weight=weight)
                data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        '''define default path to input data files.
        '''
        return ['data.csv']

    @property
    def processed_file_names(self) -> str:
        '''define default path to processed data file.
        '''
        return 'fnn_data.pt'

class CustomSubset(Subset):
    """A custom subset class that retains the 'feature' and 'y' attributes."""
    def __init__(self, dataset, indices) -> None:
        super().__init__(dataset, indices)

        self.feature = dataset.data.feature[indices]
        self.y = dataset.data.y[indices]

        if not isinstance(self.feature, torch.Tensor):
            self.feature = torch.as_tensor(self.feature)
        if not isinstance(self.y, torch.Tensor):
            self.y = torch.as_tensor(self.y)

    def __getitem__(self, idx):
        return self.feature[idx], self.y[idx]

    def __len__(self) -> int:
        return len(self.indices)
