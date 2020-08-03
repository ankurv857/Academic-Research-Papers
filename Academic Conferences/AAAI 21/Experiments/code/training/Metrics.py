import numpy as np
import torch


def wmape(y_true, y_pred, **kwargs):
    return (y_pred - y_true).abs().sum() / (y_true.sum() + 1e-6)


def mape(y_true, y_pred, replace_inf='drop', **kwargs):
    ape = (y_true - y_pred).abs()/y_true
    if replace_inf == 'ignore':
        return ape.mean()
    if isinstance(ape, torch.Tensor):
        finite_ind = torch.isfinite(ape)
        if replace_inf == 'drop':
            return ape[finite_ind].mean()
        if isinstance(replace_inf, (int, float)):
            return torch.where(finite_ind, ape, torch.tensor(replace_inf, dtype=ape.dtype))
        return torch.tensor(np.nan, dtype=ape.dtype)
    else:
        finite_ind = np.isfinite(ape)
        if replace_inf == 'drop':
            return ape[finite_ind].mean()
        if isinstance(replace_inf, (int, float)):
            return np.where(np.isfinite(ape), ape, replace_inf).mean()
        return np.nan

def binary_error(y_true, y_pred, **kwargs):
    return 1 - (y_pred * y_true + (1-y_true)*(1-y_pred)).mean()