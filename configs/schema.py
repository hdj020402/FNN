"""Type schema for model parameters — IDE autocompletion, zero runtime overhead.

Usage in main.py::

    from configs.schema import ModelParams, HparamTuningParams

    param = ModelParams.from_yaml('model_parameters.yml')
    ht_param = HparamTuningParams.from_yaml('hparam_tuning.yml')
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from typing import Any, get_type_hints


# ── Helper ────────────────────────────────────────────────────────────────────

def _dict_to_dataclass(cls: type, d: dict) -> object:
    """Recursively convert a nested dict into the given dataclass type."""
    resolved_types = get_type_hints(cls)
    field_types = {f.name: resolved_types.get(f.name, f.type) for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in d.items():
        if k not in field_types:
            continue
        target = field_types[k]
        if hasattr(target, '__dataclass_fields__') and isinstance(v, dict):
            kwargs[k] = _dict_to_dataclass(target, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


# ── Nested config classes ─────────────────────────────────────────────────────

@dataclass
class EcfpConfig:
    enabled: bool = False
    radius: int = 2
    nBits: int = 1024


@dataclass
class DescriptorsConfig:
    enabled: bool = False
    # null (or absent) → RECOMMENDED set
    # a list of descriptor names → use those exact descriptors
    # 'all_2d' → every 2-D descriptor in rdkit_descriptors.ALL_2D
    include: list[str] | str | None = None


@dataclass
class RdkitConfig:
    """RDKit molecular feature generation.

    All fields are ignored unless ``enabled`` is True.  When enabled,
    ``mol_column`` is required and ``mol_format`` specifies how to parse
    the strings in that column.
    """
    enabled: bool = False
    mol_column: str | None = None
    mol_format: str = "smiles"          # smiles | inchi
    ecfp: EcfpConfig = field(default_factory=EcfpConfig)
    descriptors: DescriptorsConfig = field(default_factory=DescriptorsConfig)


@dataclass
class DefaultFeatureConfig:
    rdkit: RdkitConfig = field(default_factory=RdkitConfig)


@dataclass
class SchedulerConfig:
    type: str = "ReduceLROnPlateau"
    factor: float = 0.7
    patience: int = 20
    min_lr: float = 0.00001


@dataclass
class EarlyStoppingConfig:
    patience: int = 50
    delta: float = 0.0


@dataclass
class ModelParams:
    """Top-level model parameters — mirrors model_parameters.yml.

    Use ``ModelParams.from_yaml(path)`` to load from a YAML file.
    """

    # ── General ───────────────────────────────────────────────────────────
    jobtype: str = "experiment"
    mode: str = "training"          # training / hpo / prediction / fine-tuning
    seed: int = 42

    # ── Dataset ───────────────────────────────────────────────────────────
    path: str = "data"
    data_file: str = "data/data.csv"
    weight_file: str | None = None
    default_feature: DefaultFeatureConfig = field(default_factory=DefaultFeatureConfig)
    feature_list: list[str] = field(default_factory=list)
    target_list: list[str] = field(default_factory=lambda: ["target1"])
    target_transform: str | None = None   # LN / LG / E^-x / null
    batch_size: int = 32
    num_workers: int = 4
    split_method: str = "random"          # random / manual
    split_file: str | None = None
    train_size: float = 0.6
    val_size: float = 0.2

    # ── Model ─────────────────────────────────────────────────────────────
    pretrained_model: str | None = None
    hidden_layer: list[int] = field(default_factory=list)
    loss_fn: str = "MSE"                  # MAE / MSE
    optimizer: str = "Adam"
    lr: float = 0.001
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # ── Training ──────────────────────────────────────────────────────────
    accumulation_step: int = 1
    epoch_num: int = 200
    output_step: int = 1
    model_save_step: int = 5
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    criteria_list: list[str] = field(default_factory=list)
    optim_criteria: str = "MSE"

    # ── Prediction ────────────────────────────────────────────────────────
    dataset_range: str = "whole"          # train / val / test / whole

    # ── Runtime (populated by main) ───────────────────────────────────────
    time: str = ""                        # timestamp set at runtime

    # ── GPU ───────────────────────────────────────────────────────────────
    GPU_memo_frac: float = 1.0
    use_deterministic: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> ModelParams:
        """Load parameters from a YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            raw: dict = yaml.full_load(f)
        return _dict_to_dataclass(cls, raw)

    def to_yaml(self, path: str) -> None:
        """Save parameters to a YAML file."""
        import dataclasses
        d = dataclasses.asdict(self)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(d, f, allow_unicode=True, sort_keys=False)


@dataclass
class ContinueTrialsConfig:
    continue_: bool = False
    storage: str | None = None
    study_name: str | None = None


@dataclass
class SamplerConfig:
    type: str = "TPESampler"
    seed: int = 42


@dataclass
class PrunerConfig:
    type: str = "MedianPruner"
    n_warmup_steps: int = 20


@dataclass
class OptunaConfig:
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    pruner: PrunerConfig = field(default_factory=PrunerConfig)
    direction: str = "minimize"
    n_trials: int = 100
    continue_trials: ContinueTrialsConfig = field(default_factory=ContinueTrialsConfig)


@dataclass
class HparamTuningParams:
    """Top-level hyperparameter tuning parameters — mirrors hparam_tuning.yml."""

    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str) -> HparamTuningParams:
        """Load HPO parameters from a YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            raw: dict = yaml.full_load(f)
        optuna_raw = raw.pop('optuna', {})
        if 'continue_trials' in optuna_raw:
            ct = optuna_raw['continue_trials']
            if 'continue' in ct:
                ct['continue_'] = ct.pop('continue')
        obj = cls()
        obj.optuna = _dict_to_dataclass(OptunaConfig, optuna_raw)
        obj._extra = raw
        return obj

    def __getitem__(self, key: str) -> Any:
        return self._extra[key]

    def __contains__(self, key: str) -> bool:
        return key in self._extra

    def get(self, key: str, default: Any = None) -> Any:
        return self._extra.get(key, default)

    def items(self):
        return self._extra.items()
