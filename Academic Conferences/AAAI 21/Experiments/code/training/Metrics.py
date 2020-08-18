import numpy as np
import torch

def BCE(y_true, y_pred, **kwargs):
    return -1 * ( (np.log(y_pred) * y_true + (1-y_true)*np.log(1-y_pred)).mean())