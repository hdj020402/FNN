from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
from typing import Dict

import torch
import os, yaml
import torch_geometric.transforms as T
import numpy as np

from datasets.datasets import CustomDataset, CustomSubset
from utils.calc_mean_std import calc_mean_std

class data_processing():
    def __init__(self, param: Dict, reprocess: bool = True) -> None:
        self.param = param
        self.path = param['path']
        self.data_file = param['data_file']
        self.weight_file = param['weight_file']
        self.sdf_file = param['sdf_file']
        self.default_feature = param['default_feature']
        self.feature_list = param['feature_list']
        self.target_list = param['target_list']
        self.transform = param['target_transform']
        self.seed = param['seed']
        self.split_method = param['split_method']
        self.split_path = param['SPLIT_file']
        self.train_size = param['train_size']
        self.val_size = param['val_size']
        self.batch_size = param['batch_size']
        self.num_workers = param['num_workers']
        self.reprocess = reprocess
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = self.gen_dataset()
        self.train_dataset, self.val_dataset, self.test_dataset, self.pred_dataset = self.split_dataset()
        self.mean, self.std = self.get_mean_std()
        self.normalization()
        self.train_dataset, self.val_dataset, self.test_dataset, self.pred_dataset = self.split_dataset()
        self.train_loader, self.val_loader, self.test_loader, self.pred_loader = self.gen_loader()

    def gen_dataset(self):
        dataset = CustomDataset(
            root = self.path,
            data_file = self.data_file,
            weight_file = self.weight_file,
            default_feature = self.default_feature,
            feature_list = self.feature_list,
            target_list = self.target_list,
            sdf_file = self.sdf_file,
            reprocess = self.reprocess
            )

        dataset = self.target_transform(dataset)

        with open(os.path.join(self.path, f'processed/model_parameters.yml'), 'w', encoding = 'utf-8') as mp:
            yaml.dump(self.param, mp, allow_unicode = True, sort_keys = False)
        return dataset

    def target_transform(self, dataset: CustomDataset) -> CustomDataset:
        if self.transform == 'LN':
            dataset.data.y = torch.log(dataset.y)
        elif self.transform == 'LG':
            dataset.data.y = torch.log10(dataset.y)
        elif self.transform == 'E^-x':
            dataset.data.y = torch.exp(-dataset.y)
        elif not self.transform:
            pass

        return dataset

    def split_dataset(self):
        train_dataset = None
        val_dataset = None
        test_dataset = None
        pred_dataset = None


        pred_dataset = self.dataset

        if self.split_method == 'random':
            train_size = int(self.train_size * len(self.dataset))
            val_size = int(self.val_size * len(self.dataset))
            test_size = len(self.dataset) - train_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset,
                [train_size, val_size, test_size],
                generator = torch.Generator().manual_seed(self.seed)
                )

        elif self.split_method == 'manual':
            indices = np.load(self.split_path, allow_pickle=True)
            train_dataset = Subset(self.dataset, indices[0])
            val_dataset = Subset(self.dataset, indices[1])
            test_dataset = Subset(self.dataset, indices[2])

        else:
            raise NotImplementedError("Split method not implemented.")

        return train_dataset, val_dataset, test_dataset, pred_dataset

    def gen_loader(self) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = True,
            pin_memory=True
            )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False,
            pin_memory=True
            )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False,
            pin_memory=True
            )
        pred_loader = DataLoader(
            self.pred_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False,
            pin_memory=True
            )

        return train_loader, val_loader, test_loader, pred_loader

    def normalization(self):
        if self.param['feature_list']:
            data_scaled = (torch.cat([self.dataset.feature, self.dataset.y], dim = 1) - self.mean) / self.std
            self.dataset.data.feature = data_scaled[:, 0:len(self.feature_list)]
            self.dataset.data.y = data_scaled[:, len(self.feature_list):len(self.feature_list) + len(self.target_list)]
        else:
            self.dataset.data.y = (self.dataset.y - self.mean) / self.std

    def get_mean_std(self):
        if self.param['mode'] == 'prediction':
            pretrained_model = self.param['pretrained_model']
            state_dict: Dict = torch.load(pretrained_model, map_location = torch.device('cpu'))
            mean = state_dict['mean']
            std = state_dict['std']
        else:
            train_dataset = CustomSubset(self.dataset, self.train_dataset.indices)
            if self.param['feature_list']:
                mean, std = calc_mean_std(
                    torch.cat(
                        [
                            train_dataset.feature,
                            train_dataset.y
                            ],
                        dim = 1
                        )
                    )
            else:
                mean, std = calc_mean_std(train_dataset.y)
        return mean, std

