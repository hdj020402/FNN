import os, shutil, json, yaml, math, logging
import torch
import optuna
from torch_geometric.loader import DataLoader
from typing import Dict, Tuple, List, Literal

from datasets.datasets import CustomDataset
from utils.time import convert_time
from utils.post_processing import read_log

def setup_logger(logger_name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

class file_processing():
    def __init__(
        self,
        param: Dict,
        ht_param: Dict | None = None,
        trial: optuna.Trial | None = None
        ) -> None:
        self.param = param
        self.TIME = self.param['time']
        self.jobtype = self.param['jobtype']
        self.ht_param = ht_param
        self.trial = trial

    def pre_make(self) -> None:
        self.subtasks = ['Overall'] + self.param['target_list']
        def make_subtask_dir(maintask_dir: str):
            for subtask in self.subtasks:
                os.makedirs(f'{maintask_dir}/{subtask}')
        self.error_dict = {}
        for subtask in self.subtasks:
            if subtask == 'Overall':
                self.error_dict[subtask] = {'LR': None, 'Loss': None, 'Train': {}, 'Val': {}, 'Test': {}}
            else:
                self.error_dict[subtask] = {'Train': {}, 'Val': {}, 'Test': {}}

        if self.param['mode'] == 'prediction':
            self.error_dict = {}
            for subtask in self.subtasks:
                self.error_dict[subtask] = {'Pred': {}}
            os.makedirs(f'Prediction_Recording/{self.jobtype}/{self.TIME}')
            with open(os.path.join(f'Prediction_Recording/{self.jobtype}/{self.TIME}', 'model_parameters.yml'), 'w', encoding = 'utf-8') as mp:
                yaml.dump(self.param, mp, allow_unicode = True, sort_keys = False)
            self.plot_dir = f'Prediction_Recording/{self.jobtype}/{self.TIME}/Plot'
            os.makedirs(self.plot_dir)
            make_subtask_dir(self.plot_dir)
            self.data_dir = f'Prediction_Recording/{self.jobtype}/{self.TIME}/Data'
            os.makedirs(self.data_dir)
            make_subtask_dir(self.data_dir)
            self.model_dir = f'Prediction_Recording/{self.jobtype}/{self.TIME}/Model'
            os.makedirs(self.model_dir)
            shutil.copy(self.param['pretrained_model'], self.model_dir)
            self.log_file = f'Prediction_Recording/{self.jobtype}/{self.TIME}/prediction_{self.TIME}.log'
            self.prediction_logger = setup_logger(f'prediction_{self.TIME}_logger', self.log_file)

        elif self.param['mode'] == 'hparam_tuning':
            os.makedirs(f'HPTuning_Recording/{self.jobtype}/{self.TIME}', exist_ok=True)
            self.optuna_log = f'HPTuning_Recording/{self.jobtype}/{self.TIME}/hptuning_{self.TIME}.log'
            self.optuna_db = f'sqlite:///HPTuning_Recording/{self.jobtype}/{self.TIME}/hptuning_{self.TIME}.db'
            self.hptuning_logger = setup_logger(f'hptuning_{self.TIME}_logger', self.optuna_log)

            if self.trial is None:
                return

            n_trials = self.ht_param['optuna']['n_trials']
            trial_name = f'Trial_{self.trial.number:0{len(str(n_trials))}d}'
            os.makedirs(f'HPTuning_Recording/{self.jobtype}/{self.TIME}/{trial_name}')
            with open(f'HPTuning_Recording/{self.jobtype}/{self.TIME}/{trial_name}/model_parameters.yml', 'w', encoding = 'utf-8') as mp:
                yaml.dump(self.param, mp, allow_unicode = True, sort_keys = False)
            if not os.path.exists(f'HPTuning_Recording/{self.jobtype}/{self.TIME}/hparam_tuning.yml'):
                with open(f'HPTuning_Recording/{self.jobtype}/{self.TIME}/hparam_tuning.yml', 'w', encoding = 'utf-8') as mp:
                    yaml.dump(self.ht_param, mp, allow_unicode = True, sort_keys = False)
            self.plot_dir =  f'HPTuning_Recording/{self.jobtype}/{self.TIME}/{trial_name}/Plot'
            os.makedirs(self.plot_dir)
            make_subtask_dir(self.plot_dir)
            self.model_dir = f'HPTuning_Recording/{self.jobtype}/{self.TIME}/{trial_name}/Model'
            os.makedirs(self.model_dir)
            self.ckpt_dir = f'HPTuning_Recording/{self.jobtype}/{self.TIME}/{trial_name}/Model/checkpoint'
            os.makedirs(self.ckpt_dir)
            self.log_file = f'HPTuning_Recording/{self.jobtype}/{self.TIME}/{trial_name}/training_{trial_name}.log'
            self.training_logger = setup_logger(f'training_{trial_name}_logger', self.log_file)

        else:
            os.makedirs(f'Training_Recording/{self.jobtype}/{self.TIME}')
            with open(os.path.join(f'Training_Recording/{self.jobtype}/{self.TIME}', 'model_parameters.yml'), 'w', encoding = 'utf-8') as mp:
                yaml.dump(self.param, mp, allow_unicode = True, sort_keys = False)
            self.plot_dir = f'Training_Recording/{self.jobtype}/{self.TIME}/Plot'
            os.makedirs(self.plot_dir)
            make_subtask_dir(self.plot_dir)
            self.model_dir = f'Training_Recording/{self.jobtype}/{self.TIME}/Model'
            os.makedirs(self.model_dir)
            self.ckpt_dir = f'Training_Recording/{self.jobtype}/{self.TIME}/Model/checkpoint'
            os.makedirs(self.ckpt_dir)
            if not os.path.isdir(f'Training_Recording/{self.jobtype}/recording'):
                os.makedirs(f'Training_Recording/{self.jobtype}/recording')
            self.log_file = f'Training_Recording/{self.jobtype}/{self.TIME}/training_{self.TIME}.log'
            self.training_logger = setup_logger(f'training_{self.TIME}_logger', self.log_file)

        self.gpu_logger = setup_logger(f'gpu_{self.TIME}_logger', f'{os.path.dirname(self.log_file)}/gpu_monitor.log')

    def basic_info_log(
        self,
        dataset: CustomDataset,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        pred_loader: DataLoader,
        mean: float,
        std: float,
        model: torch.nn.Module,
        end_time: float,
        start_time: float,
        ) -> None:
        dp_time = end_time - start_time
        hours, minutes, seconds = convert_time(dp_time)
        if self.param['mode'] == 'prediction':
            self.prediction_logger.info(f"data_path: {os.path.abspath(self.param['path'])}")
            self.prediction_logger.info(json.dumps(self.param))
            self.prediction_logger.info(f"dataset: {str(dataset.data)}")
            self.prediction_logger.info(f"size of pred set: {len(pred_loader.dataset)}")
            self.prediction_logger.info(f"batch size: {pred_loader.batch_size}")
            self.prediction_logger.info(f"mean: {mean}, std: {std}")
            self.prediction_logger.info(f"Model:\n{model}")
            self.prediction_logger.info(f"Data processing time: {hours} h {minutes} m {seconds} s")
            self.prediction_logger.info("Begin predicting...")
        else:
            self.training_logger.info(f"data_path: {os.path.abspath(self.param['path'])}")
            self.training_logger.info(json.dumps(self.param))
            self.training_logger.info(f"dataset: {str(dataset.data)}")
            self.training_logger.info(f"size of test set: {len(test_loader.dataset)}")
            self.training_logger.info(f"size of val set: {len(val_loader.dataset)}")
            self.training_logger.info(f"size of training set: {len(train_loader.dataset)}")
            self.training_logger.info(f"batch size: {train_loader.batch_size}")
            self.training_logger.info(f"mean: {mean}, std: {std}")
            self.training_logger.info(f"Model:\n{model}")
            self.training_logger.info(f"Data processing time: {hours} h {minutes} m {seconds} s")
            self.training_logger.info("Begin training...")

    def load_model(
        self,
        state_dict: Dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        mode: Literal['training', 'prediction', 'fine-tuning']
        ) -> None:
        model.load_state_dict(state_dict['model'])
        if mode == 'training':
            optimizer.load_state_dict(state_dict['optimizer'])

    def pre_train(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
        ) -> Tuple[int, Dict]:
        start_epoch = 1
        criteria_info_dict = {}
        criteria = self.param['optim_criteria']
        criteria_info_dict[criteria] = {
            'best_error': math.inf,
            'best_epoch': None,
            'best_epoch_info': None
        }
        pretrained_model = self.param['pretrained_model']
        mode = self.param['mode']
        if mode in ['prediction', 'fine-tuning']:
            state_dict: Dict = torch.load(pretrained_model, map_location=device)
            self.load_model(state_dict, model, optimizer, mode)
            start_epoch = 1
        # resume training
        elif mode == 'training':
            if pretrained_model:
                state_dict: Dict = torch.load(pretrained_model, map_location=device)
                self.load_model(state_dict, model, optimizer, mode)
                start_epoch = state_dict['epoch'] + 1
                pre_dir = os.path.dirname(os.path.dirname(os.path.dirname(pretrained_model)))
                pre_TIME = os.path.basename(pre_dir)
                pre_log_file = os.path.join(pre_dir, f'training_{pre_TIME}.log')
                shutil.copy(pre_log_file, f'Training_Recording/{self.jobtype}/{self.TIME}/pre.log')
                pre_log_info = read_log(pre_log_file, self.param)
                pre_log_text = pre_log_info.restart(start_epoch)
                with open(self.log_file, 'a') as lf:
                    lf.writelines(pre_log_text)
                pre_epoch_list: List = pre_log_info.get_performance()['Epoch'][:start_epoch - 1]
                criteria_list: List = pre_log_info.get_performance()['Overall']['Val'][criteria][:start_epoch - 1]
                criteria_info_dict[criteria]['best_error'] = min(criteria_list)
                best_index = criteria_list.index(criteria_info_dict[criteria]['best_error'])
                criteria_info_dict[criteria]['best_epoch'] = pre_epoch_list[best_index]
                criteria_info_dict[criteria]['best_epoch_info'] = pre_log_text[best_index]
        self.start_epoch = start_epoch
        return start_epoch, criteria_info_dict

    def pred_log(self, info: Dict):
        self.prediction_logger.info(json.dumps(info))

    def training_log(
        self,
        epoch: int,
        info: Dict,
        criteria_info_dict: Dict
        ):
        criteria = self.param['optim_criteria']
        if epoch % self.param['output_step'] == 0:
            self.training_logger.info(
                f'{info} '
                f'Best is epoch {criteria_info_dict[criteria]["best_epoch"]} with value'
                f': {criteria_info_dict[criteria]["best_error"]}.'
                )

    def hptuning_log(self, study: optuna.Study):
        self.hptuning_logger.info(f'best value: {study.best_value}')
        self.hptuning_logger.info(f'best params: {study.best_params}')

    def ending_log(
        self,
        criteria_info_dict: Dict,
        end_time: float,
        start_time: float,
        epoch: int
        ) -> None:
        tot_time = end_time - start_time
        epoch_time = tot_time / (epoch - self.start_epoch + 1)
        self.training_logger.info('Ending...')
        criteria = self.param['optim_criteria']
        self.training_logger.info(f"Best_{criteria}_val: {criteria_info_dict[criteria]['best_error']}")
        self.training_logger.info(f"Best_epoch_{criteria}: {criteria_info_dict[criteria]['best_epoch']}")
        self.training_logger.info(f"Best_epoch_info: {criteria_info_dict[criteria]['best_epoch_info']}")
        hours, minutes, seconds = convert_time(tot_time)
        self.training_logger.info(f'Total_time: {hours} h {minutes} m {seconds} s')
        hours, minutes, seconds = convert_time(epoch_time)
        self.training_logger.info(f'Time_per_epoch: {hours} h {minutes} m {seconds} s')
