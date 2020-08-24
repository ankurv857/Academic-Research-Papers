import numpy as np
from hyperopt import tpe, Trials, fmin, STATUS_OK, hp, space_eval
from functools import partial
from hyperopt.pyll.base import scope
from operator import itemgetter


class HyperOpt:
    def __init__(self, model, method, param_grid, config, model_config, **kwargs):
        self.model = model
        self.method = method
        self.config = config
        self.model_config = model_config
        if method == 'hyperopt':
            param_grid = get_hyperopt_param_grid(param_grid)
        self.param_grid = param_grid
        self.trials = 0

    def _objective(self, cv_data, params):
        self.trials += 1
        self.model.set_params(**params)
        metrics_dict = self.model.cross_validate(cv_data, self.config , self.model_config)
        loss = []
        for penalty,  metrics in enumerate(metrics_dict['cv_data']['model_metrics']):
            loss.append(metrics['val']/(penalty + 1))
        loss = np.sum(loss)
        result = dict(trial = self.trials, loss = loss, status = STATUS_OK, params = params)
        return result

    def optimize(self, cv_data, max_evals=100, **kwargs):
        if self.method == 'hyperopt':
            trials = Trials()
            fmin(partial(self._objective, cv_data), self.param_grid, tpe.suggest, max_evals, trials=trials)
            trials_results = sorted(trials.results, key=itemgetter('loss'), reverse=False)
            return trials_results[:self.model_config.best_trials]


def get_hyperopt_param_grid(param_grid):
    """

        Args:
            param_grid: parameters for the ML models

        Returns:

    """

    hp_param_grid = {}
    for key, val in param_grid.items():
        if val['type'] == 'choice':
            hp_param_grid[key] = hp.choice(key, val['params'])
        elif val['type'] == 'quniform':
            hp_param_grid[key] = scope.int(hp.quniform(key, *val['params']))
        elif val['type'] == 'uniform':
            hp_param_grid[key] = hp.uniform(key, *val['params'])
        elif val['type'] == 'loguniform':
            hp_param_grid[key] = hp.loguniform(key, *np.log(val['params']))

    return hp_param_grid