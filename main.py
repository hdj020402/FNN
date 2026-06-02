"""FNN — Feedforward Neural Network toolkit for molecular property prediction.

Entry point. Behaviour is determined by ``param.mode``:

    python main.py               # training (reads model_parameters.yml)
    python main.py               # hpo (mode=hpo, reads hparam_tuning.yml)
    python main.py               # prediction (mode=prediction)
"""
import os
import time
import json
import torch
import optuna
import pandas as pd
import numpy as np
from copy import deepcopy
from functools import partial

from configs.schema import ModelParams, HparamTuningParams
from data.data_processing import DataProcessing
from utils.gen_model import gen_model, gen_optimizer, gen_scheduler
from utils.setup_seed import setup_seed
from utils.visualization import scatter, scatterFromModel, loss_epoch
from utils.evaluation import Evaluation
from utils.metrics import Metrics
from utils.optuna_setup import OptunaSetup
from utils.train import train, validate
from utils.file_processing import FileProcessing, LogParser
from utils.save_model import SaveModel
from utils.timer import Timer
from utils.gpu_monitor import GPUMonitor
from utils.utils import extract_keys_and_lists


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _kfold_indices(n_samples: int, n_folds: int, shuffle: bool = True,
                   seed: int = 42) -> list[tuple[list[int], list[int]]]:
    """Generate k-fold train/val index pairs without sklearn dependency."""
    indices = list(range(n_samples))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    fold_size = n_samples // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n_samples
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, val_idx))
    return folds


# ═══════════════════════════════════════════════════════════════════════════════
# HPO helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _flatten_search_space(d: dict, prefix: str = '') -> dict:
    """Recursively flatten nested search-space dict to ``{dotted_path: spec}``."""
    result = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and 'type' not in v:
            result.update(_flatten_search_space(v, key))
        else:
            result[key] = v
    return result


def _suggest(trial: optuna.Trial, name: str, spec: dict):
    """Suggest a hyperparameter value based on spec type."""
    stype = spec['type']
    kw = {k: v for k, v in spec.items() if k != 'type'}
    if stype == 'int':
        return trial.suggest_int(name, **kw)
    elif stype == 'float':
        return trial.suggest_float(name, **kw)
    elif stype == 'loguniform':
        return trial.suggest_float(name, kw['low'], kw['high'], log=True)
    elif stype == 'discrete_uniform':
        return trial.suggest_float(name, kw['low'], kw['high'], step=kw['q'])
    elif stype == 'categorical':
        return trial.suggest_categorical(name, kw['choices'])
    else:
        raise ValueError(f"Unknown suggest type '{stype}' for param '{name}'")


def _set_dot_path(obj, dotted_path: str, value) -> None:
    """Set a nested attribute via dot-notation key (e.g. ``'training.lr'``)."""
    parts = dotted_path.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _handle_hidden_layer(trial: optuna.Trial, attr: dict) -> list[int]:
    """Suggest a hidden layer configuration for HPO."""
    num_layers = _suggest(trial, 'num_layers', attr['num_layers'])
    return [_suggest(trial, f'neuron_{i}', attr['neuron']) for i in range(num_layers)]


# ═══════════════════════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════════════════════

def training(param: ModelParams, ht_param: HparamTuningParams | None = None,
             trial: optuna.Trial | None = None,
             fold_indices: tuple[list[int], list[int]] | None = None,
             output_subdir: str | None = None) -> float:
    """Run training (or a single HPO trial, or a single CV fold).

    Args:
        param: Model configuration.
        ht_param: HPO configuration (only for HPO mode).
        trial: Optuna trial (only inside an HPO study).
        fold_indices: ``(train_idx, val_idx)`` for one CV fold.  ``None`` for
            normal (non-CV) training.
        output_subdir: Subdirectory under the training output root (e.g.
            ``'fold_0'``).  ``None`` for normal training.

    Returns:
        Best validation loss.
    """
    fp = FileProcessing(param, ht_param, trial, output_subdir=output_subdir)
    fp.pre_make()
    plot_dir, model_dir, ckpt_dir = fp.plot_dir, fp.model_dir, fp.ckpt_dir
    training_logger = fp.training_logger
    gpu_monitor = GPUMonitor(fp.gpu_logger)
    gpu_monitor.start()
    error_dict = fp.error_dict

    setup_seed(param.seed, param.use_deterministic)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if param.use_deterministic:
        torch.use_deterministic_algorithms(True)

    epoch_num = param.epoch_num

    dp_timer = Timer()
    dp_timer.start()
    dp = DataProcessing(param, fold_indices=fold_indices)
    dataset = dp.dataset
    norm_dict = dp.norm_dict
    train_loader = dp.train_loader
    val_loader = dp.val_loader
    test_loader = dp.test_loader
    dp_timer.end()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gen_model(param, dataset)
    optimizer = gen_optimizer(param, model)
    scheduler = gen_scheduler(param, optimizer)

    fp.basic_info_log(
        dataset, train_loader, val_loader, test_loader, None,
        norm_dict, model, dp_timer,
    )

    criteria_set = set(list(param.criteria_list) + [param.loss_fn])

    eval_class = partial(
        Evaluation,
        device=device,
        norm_dict=norm_dict,
        transform=param.target_transform,
    )

    model_saving = SaveModel(norm_dict, param, model_dir, ckpt_dir, training_logger.info)
    start_epoch = fp.pre_train(model, optimizer, device, model_saving)

    timer = Timer()
    timer.start()
    for epoch in range(start_epoch, epoch_num + 1):
        try:
            lr = scheduler.optimizer.param_groups[0]['lr'] if scheduler else param.lr
            loss = train(
                model, train_loader, optimizer, param.loss_fn, device,
                param.accumulation_step,
            )
            validate(model, val_loader, param.loss_fn, device)
            error_dict['Overall']['LR'] = round(lr, 7)
            error_dict['Overall']['Loss'] = round(loss, 7)

            for phase, loader in zip(
                ['Train', 'Val', 'Test'],
                [train_loader, val_loader, test_loader],
            ):
                evaluation = eval_class(loader, model)
                pred, target = evaluation.pred, evaluation.target
                err = Metrics(pred, target)
                for criteria in criteria_set:
                    errors = torch.cat([
                        getattr(err, criteria)(dim=None).unsqueeze(0),
                        getattr(err, criteria)(dim=0).view(-1),
                    ])
                    for subtask, error in zip(error_dict.keys(), errors):
                        error_dict[subtask][phase][criteria] = round(float(error), 7)

            if scheduler is not None:
                scheduler.step(error_dict['Overall']['Val'][param.loss_fn])

            info = json.dumps({'Epoch': epoch} | error_dict)

            model_saving.best_model(model, optimizer, epoch,
                                    error_dict['Overall']['Val'][param.optim_criteria])
            model_saving.regular_model(model, optimizer, epoch)
            fp.training_log(epoch, info, model_saving.best_val_loss, model_saving.best_epoch)
            torch.cuda.empty_cache()

            if trial is not None:
                trial.report(error_dict['Overall']['Val'][param.optim_criteria], epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if model_saving.check_early_stopping():
                break

        except torch.cuda.OutOfMemoryError as e:
            training_logger.error(e)
            break
    timer.end()

    fp.ending_log(timer, epoch)
    gpu_monitor.stop()

    # Scatter plots from best model
    scatterFromModel(
        f"{model_dir}/best_model_{param.optim_criteria}_{param.time}.pth",
        param, dp, plot_dir,
    )

    # Loss-epoch curves
    log_parser = LogParser(fp.log_file)
    log_info_dict = log_parser.get_performance()
    info_pairs = extract_keys_and_lists(log_info_dict)
    for item, data in info_pairs:
        if item == 'Epoch':
            continue
        loss_epoch(
            [[log_info_dict['Epoch'], data]],
            [f'{item}-Epoch'],
            ['#03658C'],
            'Epoch', f'{item}',
            f'{plot_dir}/{item.split("_")[0]}/{item}-Epoch_{param.time}.png',
        )
        pd.DataFrame({'Epoch': log_info_dict['Epoch'], item: data}).to_csv(
            f'{plot_dir}/{item.split("_")[0]}/{item}-Epoch_{param.time}.csv',
            index=False,
        )

    return model_saving.best_val_loss


def cross_validation(param: ModelParams) -> None:
    """Run k-fold cross-validation.

    For each fold the model is trained from scratch with its own
    normalisation.  Results are aggregated in the common output directory.

    Args:
        param: Model configuration (``n_folds`` must be > 1).
    """
    n_folds = param.n_folds

    # ── Determine dataset size (streaming line count — no CSV parsing) ──
    with open(param.data_file, encoding='utf-8') as f:
        n_samples = sum(1 for _ in f) - 1  # header row

    if n_folds > n_samples:
        raise ValueError(
            f"n_folds ({n_folds}) cannot exceed the number of samples "
            f"({n_samples})."
        )

    folds = _kfold_indices(n_samples, n_folds, shuffle=True, seed=param.seed)

    # ── Output setup ──────────────────────────────────────────────────────
    summary_logger = FileProcessing(param).setup_cv_summary(n_samples)

    fold_results: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        summary_logger.info(f'--- Fold {fold + 1}/{n_folds} ---')
        fold_param = deepcopy(param)

        result = training(fold_param, fold_indices=(train_idx, val_idx),
                          output_subdir=f'fold_{fold}')
        fold_results.append(result)

        summary_logger.info(
            f'Fold {fold + 1} best {param.optim_criteria}: {result:.7f}'
        )

    # ── Summary ───────────────────────────────────────────────────────────
    arr = np.array(fold_results)
    summary_logger.info('=== CV Summary ===')
    for i, v in enumerate(fold_results):
        summary_logger.info(f'  Fold {i + 1}: {v:.7f}')
    summary_logger.info(f'  Mean  : {arr.mean():.7f}')
    summary_logger.info(f'  Std   : {arr.std():.7f}')
    summary_logger.info(f'  Min   : {arr.min():.7f}')
    summary_logger.info(f'  Max   : {arr.max():.7f}')


def prediction(param: ModelParams) -> None:
    """Run prediction using a pretrained model.

    Args:
        param: Model configuration (mode must be 'prediction').
    """
    fp = FileProcessing(param)
    fp.pre_make()
    plot_dir, data_dir = fp.plot_dir, fp.data_dir
    gpu_monitor = GPUMonitor(fp.gpu_logger)
    gpu_monitor.start()
    subtasks = fp.subtasks
    error_dict = fp.error_dict

    setup_seed(param.seed, param.use_deterministic)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if param.use_deterministic:
        torch.use_deterministic_algorithms(True)

    dp_timer = Timer()
    dp_timer.start()
    dp = DataProcessing(param)
    dataset = dp.dataset
    norm_dict = dp.norm_dict
    train_loader = dp.train_loader
    val_loader = dp.val_loader
    test_loader = dp.test_loader
    pred_loader = dp.pred_loader
    dp_timer.end()

    loader_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader, 'whole': pred_loader}
    try:
        loader = loader_dict[param.dataset_range]
    except KeyError:
        loader = pred_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gen_model(param, dataset)
    optimizer = gen_optimizer(param, model)
    fp.pre_train(model, optimizer, device)

    fp.basic_info_log(
        dataset, None, None, None, loader,
        norm_dict, model, dp_timer,
    )

    criteria_list = param.criteria_list

    evaluation = Evaluation(loader, model, device, norm_dict, param.target_transform)
    pred, target = evaluation.pred, evaluation.target
    gpu_monitor.stop()

    for criteria in criteria_list:
        errors = torch.cat([
            getattr(Metrics(pred, target), criteria)(dim=None).unsqueeze(0),
            getattr(Metrics(pred, target), criteria)(dim=0).view(-1),
        ])
        for subtask, error in zip(error_dict.keys(), errors):
            error_dict[subtask]['Pred'][criteria] = round(float(error), 7)
    fp.pred_log(error_dict)

    for subtask, idx in zip(subtasks, range(-1, len(param.target_list))):
        if subtask == 'Overall':
            torch.save(pred, f'{data_dir}/{subtask}/pred.pt')
            torch.save(target, f'{data_dir}/{subtask}/target.pt')
        else:
            torch.save(pred[:, idx], f'{data_dir}/{subtask}/pred.pt')
            torch.save(target[:, idx], f'{data_dir}/{subtask}/target.pt')

    for t, p, task in zip(
        torch.split(target, 1, dim=-1),
        torch.split(pred, 1, dim=-1),
        param.target_list,
    ):
        scatter(
            [t, p],
            scatter_label=['eval'],
            output_path=f"{plot_dir}/{task}/{param.dataset_range}.png",
        )


def hpo(param: ModelParams, ht_param: HparamTuningParams) -> None:
    """Run Optuna hyperparameter optimization.

    Args:
        param: Base model configuration.
        ht_param: HPO configuration with search space.
    """
    fp = FileProcessing(param, ht_param)
    fp.pre_make()
    storage_name = fp.optuna_db

    setup_seed(param.seed, param.use_deterministic)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if param.use_deterministic:
        torch.use_deterministic_algorithms(True)

    # Set up Optuna logging
    optuna_logger = __import__('logging').getLogger('optuna')
    optuna_logger.handlers = []
    optuna_logger.addHandler(fp.hpo_logger.handlers[0])
    optuna_logger.setLevel(__import__('logging').INFO)
    optuna_logger.propagate = False

    search_space_raw = dict(ht_param.items())
    search_space = _flatten_search_space(search_space_raw)

    def objective(trial: optuna.Trial) -> float:
        trial_param = deepcopy(param)
        for dotted_key, spec in search_space.items():
            if dotted_key == 'hidden_layer':
                trial_param.hidden_layer = _handle_hidden_layer(trial, spec)
            else:
                _set_dot_path(trial_param, dotted_key, _suggest(trial, dotted_key, spec))
        return training(trial_param, ht_param, trial)

    optuna_setup = OptunaSetup(param, ht_param)
    optuna_setup.logging_setup(fp.hpo_logger)
    study = optuna_setup.create_study(f'hpo_{param.jobtype}', storage_name)
    study.optimize(objective, n_trials=ht_param.optuna.n_trials)
    fp.hpo_log(study)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    TIME = time.strftime('%b_%d_%Y_%H%M%S', time.localtime())

    param = ModelParams.from_yaml('configs/model_parameters.yaml')
    param.time = TIME

    setup_seed(param.seed, param.use_deterministic)
    if param.use_deterministic:
        torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available() and param.GPU_memo_frac < 1.0:
        torch.cuda.set_per_process_memory_fraction(param.GPU_memo_frac)

    if param.mode in ['training', 'fine-tuning']:
        if param.n_folds > 1:
            cross_validation(param)
        else:
            training(param)
    elif param.mode == 'hpo':
        ht_param = HparamTuningParams.from_yaml('configs/hpo.yaml')
        hpo(param, ht_param)
    elif param.mode == 'prediction':
        prediction(param)
    else:
        raise ValueError(
            f"Invalid mode '{param.mode}'. Valid: training / hpo / "
            "prediction / fine-tuning"
        )
