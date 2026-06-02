"""File processing — output directory management, logging, training resume."""
import os
import json
import shutil
import logging
import socket
import torch
import optuna
from torch.utils.data import DataLoader
from typing import Literal
from copy import deepcopy

from configs.schema import ModelParams, HparamTuningParams
from data.dataset import FeatureDataset
from utils.save_model import SaveModel
from utils.timer import Timer


# Config keys included in run-record files.
_RECORD_KEYS = [
    'mode', 'seed', 'use_deterministic', 'GPU_memo_frac',
    'pretrained_model', 'time', 'jobtype',
    'path', 'sdf_file', 'data_file', 'weight_file',
    'feature_list', 'target_list', 'target_transform',
    'batch_size', 'num_workers', 'split_method', 'split_file',
    'train_size', 'val_size', 'dataset_range',
    'hidden_layer', 'loss_fn', 'optimizer', 'lr', 'scheduler',
    'accumulation_step', 'epoch_num', 'output_step', 'model_save_step',
    'early_stopping', 'criteria_list', 'optim_criteria',
]


def _setup_logger(logger_name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes to a file."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    file_handler = logging.FileHandler(log_file, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class LogParser:
    """Reads a training log file to support training resumption."""

    def __init__(self, log_file: str) -> None:
        self.log_file = log_file

    def restart(self, start_epoch: int) -> list[str]:
        """Return log lines for epochs before start_epoch (used when resuming)."""
        with open(self.log_file) as lf:
            text = lf.readlines()
        pre_log_text: list[str] = []
        i = 1
        for line in text:
            if i == start_epoch:
                break
            if '"Epoch"' in line:
                pre_log_text.append(line)
                i += 1
        return pre_log_text

    def get_performance(self) -> dict:
        """Parse log and return merged performance dict (for loss-epoch plots)."""
        import re
        from utils.utils import recursive_merge
        with open(self.log_file) as lf:
            text = lf.readlines()
        pattern = r'\{.*\}'
        dicts: list[dict] = []
        for line in text:
            if 'EarlyStopping' in line or 'Ending...' in line:
                break
            if '"Epoch"' not in line:
                continue
            match = re.search(pattern, line)
            if match:
                dicts.append(json.loads(match.group(0)))
        return recursive_merge(dicts)


class FileProcessing:
    """Manages output directories, logging, model loading, and training resumption.

    Args:
        param: ModelParams configuration.
        ht_param: HparamTuningParams (only for HPO mode).
        trial: Optuna trial (only within an HPO trial).
    """

    def __init__(
        self,
        param: ModelParams,
        ht_param: HparamTuningParams | None = None,
        trial: optuna.Trial | None = None,
        output_subdir: str | None = None,
    ) -> None:
        self.param = param
        self.TIME = param.time
        self.jobtype = param.jobtype
        self.ht_param = ht_param
        self.trial = trial
        self._output_subdir = output_subdir

    def pre_make(self) -> None:
        """Create output directory structure and set up loggers."""
        mode = self.param.mode
        self.subtasks = ['Overall'] + list(self.param.target_list)

        def make_subtask_dir(maintask_dir: str):
            for subtask in self.subtasks:
                os.makedirs(f'{maintask_dir}/{subtask}', exist_ok=True)

        # ── Error dict ────────────────────────────────────────────────────
        self.error_dict: dict = {}
        for subtask in self.subtasks:
            if subtask == 'Overall':
                self.error_dict[subtask] = {'LR': None, 'Loss': None, 'Train': {}, 'Val': {}, 'Test': {}}
            else:
                self.error_dict[subtask] = {'Train': {}, 'Val': {}, 'Test': {}}

        # ── Compute base directory ────────────────────────────────────────
        if mode == 'prediction':
            self._base_dir = f'outputs/prediction/{self.jobtype}/{self.TIME}'
        elif mode == 'hpo':
            self._base_dir = f'outputs/hpo/{self.jobtype}/{self.TIME}'
        else:
            self._base_dir = f'outputs/training/{self.jobtype}/{self.TIME}'
        if self._output_subdir:
            self._base_dir = f'{self._base_dir}/{self._output_subdir}'

        # ── Build per-mode structure ──────────────────────────────────────
        if mode == 'prediction':
            self.error_dict = {}
            for subtask in self.subtasks:
                self.error_dict[subtask] = {'Pred': {}}
            os.makedirs(self._base_dir, exist_ok=True)
            self.param.to_yaml(f'{self._base_dir}/model_parameters.yml')
            self.plot_dir = f'{self._base_dir}/plot'
            os.makedirs(self.plot_dir, exist_ok=True)
            make_subtask_dir(self.plot_dir)
            self.data_dir = f'{self._base_dir}/data'
            os.makedirs(self.data_dir, exist_ok=True)
            make_subtask_dir(self.data_dir)
            self.model_dir = f'{self._base_dir}/model'
            os.makedirs(self.model_dir, exist_ok=True)
            if self.param.pretrained_model:
                shutil.copy(self.param.pretrained_model, self.model_dir)
            self.log_file = f'{self._base_dir}/prediction_{self.TIME}.log'
            self.prediction_logger = _setup_logger(f'prediction_{self.TIME}_logger', self.log_file)

        elif mode == 'hpo':
            os.makedirs(self._base_dir, exist_ok=True)
            self.optuna_log = f'{self._base_dir}/hpo_{self.TIME}.log'
            self.optuna_db = f'sqlite:///{self._base_dir}/hpo_{self.TIME}.db'
            self.hpo_logger = _setup_logger(f'hpo_{self.TIME}_logger', self.optuna_log)

            if self.trial is None:
                return

            n_trials = self.ht_param.optuna.n_trials if self.ht_param else 100
            trial_name = f'Trial_{self.trial.number:0{len(str(n_trials))}d}'
            trial_dir = f'{self._base_dir}/{trial_name}'
            os.makedirs(trial_dir, exist_ok=True)
            self.param.to_yaml(f'{trial_dir}/model_parameters.yml')
            if not os.path.exists(f'{self._base_dir}/hparam_tuning.yml') and self.ht_param is not None:
                import yaml
                with open(f'{self._base_dir}/hparam_tuning.yml', 'w', encoding='utf-8') as f:
                    yaml.dump(self.ht_param._extra, f, allow_unicode=True, sort_keys=False)
            self.plot_dir = f'{trial_dir}/plot'
            os.makedirs(self.plot_dir, exist_ok=True)
            make_subtask_dir(self.plot_dir)
            self.model_dir = f'{trial_dir}/model'
            os.makedirs(self.model_dir, exist_ok=True)
            self.ckpt_dir = f'{trial_dir}/model/checkpoint'
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.log_file = f'{trial_dir}/training_{trial_name}.log'
            self.training_logger = _setup_logger(f'training_{trial_name}_logger', self.log_file)

        else:  # training / fine-tuning
            os.makedirs(self._base_dir, exist_ok=True)
            self.param.to_yaml(f'{self._base_dir}/model_parameters.yml')
            self.plot_dir = f'{self._base_dir}/plot'
            os.makedirs(self.plot_dir, exist_ok=True)
            make_subtask_dir(self.plot_dir)
            self.model_dir = f'{self._base_dir}/model'
            os.makedirs(self.model_dir, exist_ok=True)
            self.ckpt_dir = f'{self._base_dir}/model/checkpoint'
            os.makedirs(self.ckpt_dir, exist_ok=True)
            recording_dir = f'outputs/training/{self.jobtype}/recording'
            if not os.path.isdir(recording_dir):
                os.makedirs(recording_dir, exist_ok=True)
            self.log_file = f'{self._base_dir}/training_{self.TIME}.log'
            self.training_logger = _setup_logger(f'training_{self.TIME}_logger', self.log_file)

        self.gpu_logger = _setup_logger(
            f'gpu_{self.TIME}_logger',
            f'{os.path.dirname(self.log_file)}/gpu_monitor.log',
        )

    # ── Parameter counting ─────────────────────────────────────────────────

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
        """Count total and trainable parameters in a model.

        Returns:
            (total_params, trainable_params)
        """
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    @staticmethod
    def format_parameters(num_params: int) -> str:
        """Format parameter count as human-readable string (K, M, B)."""
        if num_params >= 1e9:
            return f"{num_params / 1e9:.2f}B"
        elif num_params >= 1e6:
            return f"{num_params / 1e6:.2f}M"
        elif num_params >= 1e3:
            return f"{num_params / 1e3:.2f}K"
        else:
            return str(num_params)

    # ── Logging ────────────────────────────────────────────────────────────

    def basic_info_log(
        self,
        dataset: FeatureDataset,
        train_loader: DataLoader | None,
        val_loader: DataLoader | None,
        test_loader: DataLoader | None,
        pred_loader: DataLoader | None,
        norm_dict: dict[str, tuple[torch.Tensor, torch.Tensor]],
        model: torch.nn.Module,
        timer: Timer,
    ) -> None:
        """Log basic dataset/model information before training/prediction.

        Args:
            dataset: The FeatureDataset instance.
            train_loader: Training DataLoader (None for prediction).
            val_loader: Validation DataLoader (None for prediction).
            test_loader: Test DataLoader (None for prediction).
            pred_loader: Prediction DataLoader.
            norm_dict: Normalization parameters.
            model: The model.
            timer: Timer that has been started/ended for data processing.
        """
        days, hours, minutes, seconds = timer.get_tot_time()
        total_params, trainable_params = self.count_parameters(model)
        total_str = self.format_parameters(total_params)
        trainable_str = self.format_parameters(trainable_params)

        if self.param.mode == 'prediction':
            logger = self.prediction_logger
            logger.info(f"hostname: {socket.gethostname()}")
            logger.info(f"data_path: {os.path.abspath(self.param.path)}")
            logger.info(f"config saved to: {self._base_dir}/model_parameters.yml")
            logger.info(f"num features: {dataset.num_features}, num targets: {dataset.num_targets}")
            logger.info(f"dataset size: {len(dataset)}")
            logger.info(f"size of pred set: {len(pred_loader.dataset)}")
            logger.info(f"batch size: {pred_loader.batch_size}")
            logger.info(f"norm info: {norm_dict}")
            logger.info(f"Model:\n{model}")
            logger.info(f"Total parameters: {total_str} ({total_params:,})")
            logger.info(f"Trainable parameters: {trainable_str} ({trainable_params:,})")
            logger.info(f"Data processing time: {days} d {hours} h {minutes} m {seconds} s")
            logger.info("Begin predicting...")
        else:
            logger = self.training_logger
            logger.info(f"hostname: {socket.gethostname()}")
            logger.info(f"data_path: {os.path.abspath(self.param.path)}")
            logger.info(f"config saved to: {self._base_dir}/model_parameters.yml")
            logger.info(f"num features: {dataset.num_features}, num targets: {dataset.num_targets}")
            logger.info(f"dataset size: {len(dataset)}")
            logger.info(f"size of test set: {len(test_loader.dataset) if test_loader else 'N/A (CV)'}")
            logger.info(f"size of val set: {len(val_loader.dataset)}")
            logger.info(f"size of training set: {len(train_loader.dataset)}")
            logger.info(f"batch size: {train_loader.batch_size}")
            logger.info(f"norm info: {norm_dict}")
            logger.info(f"Model:\n{model}")
            logger.info(f"Total parameters: {total_str} ({total_params:,})")
            logger.info(f"Trainable parameters: {trainable_str} ({trainable_params:,})")
            logger.info(f"Data processing time: {days} d {hours} h {minutes} m {seconds} s")
            logger.info("Begin training...")

    # ── Model loading / resume ─────────────────────────────────────────────

    @staticmethod
    def load_model(
        state_dict: dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        mode: Literal['training', 'prediction', 'fine-tuning'],
    ) -> None:
        """Load model/optimizer state from a checkpoint dict.

        Args:
            state_dict: Checkpoint dict with 'model' and optionally 'optimizer' keys.
            model: Model to load weights into.
            optimizer: Optimizer to load state into (only for training mode).
            mode: 'training', 'prediction', or 'fine-tuning'.
        """
        model.load_state_dict(state_dict['model'])
        if mode == 'training':
            optimizer.load_state_dict(state_dict['optimizer'])

    def pre_train(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        model_saving: SaveModel | None = None,
    ) -> int:
        """Load pretrained model or resume from checkpoint, returning start_epoch.

        Args:
            model: Model to load weights into.
            optimizer: Optimizer to load state into.
            device: Torch device.
            model_saving: SaveModel instance (for restoring best model state on resume).

        Returns:
            The epoch number to start training from (1-indexed).
        """
        start_epoch = 1
        pretrained_model = self.param.pretrained_model
        mode = self.param.mode

        if mode in ['prediction', 'fine-tuning']:
            state_dict: dict = torch.load(pretrained_model, map_location=device, weights_only=False)
            self.load_model(state_dict, model, optimizer, mode)
            start_epoch = 1

        elif mode == 'training':
            if pretrained_model:
                state_dict: dict = torch.load(pretrained_model, map_location=device, weights_only=False)
                self.load_model(state_dict, model, optimizer, mode)
                start_epoch = state_dict['epoch'] + 1
                pre_dir = os.path.dirname(os.path.dirname(os.path.dirname(pretrained_model)))
                pre_TIME = os.path.basename(pre_dir)
                pre_log_file = os.path.join(pre_dir, f'training_{pre_TIME}.log')
                if os.path.exists(pre_log_file):
                    shutil.copy(pre_log_file, f'outputs/training/{self.jobtype}/{self.TIME}/pre.log')
                    pre_log_info = LogParser(pre_log_file)
                    pre_log_text = pre_log_info.restart(start_epoch)
                    with open(self.log_file, 'a') as lf:
                        lf.writelines(pre_log_text)

                # Try to restore best model state
                pre_best_path = f'{pre_dir}/model/best_model_{pre_TIME}.pth'
                if os.path.exists(pre_best_path) and model_saving is not None:
                    best_model = deepcopy(model)
                    best_optimizer = deepcopy(optimizer)
                    pre_best_model: dict = torch.load(pre_best_path, map_location=device, weights_only=False)
                    self.load_model(pre_best_model, best_model, best_optimizer, mode)
                    model_saving.best_model(
                        best_model, best_optimizer,
                        pre_best_model['epoch'], pre_best_model['val_loss'],
                    )

        self.start_epoch = start_epoch
        return start_epoch

    # ── Logging helpers ────────────────────────────────────────────────────

    def pred_log(self, info: dict) -> None:
        """Log prediction results."""
        self.prediction_logger.info(json.dumps(info))

    def training_log(
        self,
        epoch: int,
        info: dict,
        best_val_loss: float,
        best_epoch: int,
    ) -> None:
        """Log training progress at each output_step."""
        self.best_val_loss = best_val_loss
        self.best_epoch = best_epoch
        if epoch % self.param.output_step == 0:
            self.training_logger.info(
                f'{info} '
                f'Best is epoch {best_epoch} with value: {best_val_loss}.'
            )

    def hpo_log(self, study: optuna.Study) -> None:
        """Log HPO results."""
        self.hpo_logger.info(f'best value: {study.best_value}')
        self.hpo_logger.info(f'best params: {study.best_params}')

    def ending_log(
        self,
        timer: Timer,
        epoch: int,
    ) -> None:
        """Log training summary at the end."""
        self.training_logger.info('Ending...')
        self.training_logger.info(f"Best val loss: {self.best_val_loss}")
        self.training_logger.info(f"Best epoch: {self.best_epoch}")
        days, hours, minutes, seconds = timer.get_tot_time()
        self.training_logger.info(f'Total time: {days} d {hours} h {minutes} m {seconds} s')
        days, hours, minutes, seconds = timer.get_average_time(epoch - self.start_epoch + 1)
        self.training_logger.info(f'Time per epoch: {days} d {hours} h {minutes} m {seconds} s')
