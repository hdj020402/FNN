"""Optuna study factory."""
import optuna
import logging

from configs.schema import ModelParams, HparamTuningParams


class OptunaSetup:
    """Create and configure an Optuna study from HPO configuration.

    Args:
        param: ModelParams configuration.
        ht_param: HparamTuningParams configuration.
    """

    def __init__(self, param: ModelParams, ht_param: HparamTuningParams) -> None:
        self.param = param
        self.ht_param = ht_param

    def create_pruner(self) -> optuna.pruners.BasePruner:
        """Create an Optuna pruner from config."""
        pruner_cfg = self.ht_param.optuna.pruner
        pruner = getattr(optuna.pruners, pruner_cfg.type)
        kwargs = {'n_warmup_steps': pruner_cfg.n_warmup_steps}
        return pruner(**kwargs)

    def create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create an Optuna sampler from config."""
        sampler_cfg = self.ht_param.optuna.sampler
        sampler = getattr(optuna.samplers, sampler_cfg.type)
        kwargs = {'seed': sampler_cfg.seed}
        return sampler(**kwargs)

    def create_study(self, study_name: str, storage: str) -> optuna.Study:
        """Create or resume an Optuna study.

        Args:
            study_name: Name for the study.
            storage: SQLite storage URL.

        Returns:
            An Optuna Study instance.
        """
        ct = self.ht_param.optuna.continue_trials
        if ct.continue_:
            return self._load_study()

        return optuna.create_study(
            sampler=self.create_sampler(),
            pruner=self.create_pruner(),
            direction=self.ht_param.optuna.direction,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )

    def _load_study(self) -> optuna.Study:
        """Resume an existing study."""
        ct = self.ht_param.optuna.continue_trials
        storage = ct.storage
        study_name = ct.study_name
        if study_name is None:
            summaries = optuna.get_all_study_summaries(storage=storage)
            study_name = summaries[0].study_name
        return optuna.load_study(study_name=study_name, storage=storage)

    @staticmethod
    def logging_setup(hpo_logger: logging.Logger) -> None:
        """Route Optuna's own logger into the job's training logger."""
        optuna_logger = logging.getLogger('optuna')
        optuna_logger.handlers = []
        optuna_logger.addHandler(hpo_logger.handlers[0])
        optuna_logger.setLevel(logging.INFO)
        optuna_logger.propagate = False
