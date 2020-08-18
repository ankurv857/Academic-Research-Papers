import torch.utils.data
import numpy as np
import pandas as pd


class MLPDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None):
        super().__init__()
        self.X = X
        if y is None:
            y = pd.Series(0, index=X.index)
        self.y = y

        self.emb_cols = X.select_dtypes(exclude='float64').columns
        self.cont_cols = list(set(X.columns) - set(self.emb_cols))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        ind = self.X.index[item] 
        row = self.X.loc[ind]
        return (row[self.cont_cols].values.astype(np.float32),
                row[self.emb_cols].values.astype(float).astype(int),
                self.y.loc[ind].astype(int))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None, static_cols=None, seq_len=None, feature_groups = None):
        super().__init__()
        self.X = X
        if y is None:
            y = pd.Series(0, index=X.index)
        self.y = y
        self.seq_len = seq_len

        feature_groups = feature_groups or {}
        static_cols = static_cols or []
        emb_cols = X.columns.intersection(feature_groups.get('label_encoded', []))
        self.emb_static_cols = list(set(emb_cols) & set(static_cols))
        self.emb_temporal_cols = list(set(emb_cols) - set(static_cols))
        self.cont_static_cols = list((set(X.columns) & set(static_cols)) - set(emb_cols))
        self.cont_temporal_cols = list(set(X.columns) - set(emb_cols) - set(static_cols))

        self.ts_levels = X.index.levels[-1]
        ts_levels = self.ts_levels[:len(self.ts_levels) - self.seq_len + 1]
        self.index = X.index[X.index.get_level_values(-1).isin(ts_levels)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        ind = self.index[item]
        ts_ind = self.ts_levels.tolist().index(ind[-1])
        rows = self.X.loc[ind[:-1]].loc[self.ts_levels[ts_ind: ts_ind + self.seq_len]]
        y = self.y.loc[ind[:-1]].loc[self.ts_levels[ts_ind: ts_ind + self.seq_len]]
        return (rows[self.cont_static_cols].values.astype(np.float32),
                rows[self.cont_temporal_cols].values.astype(np.float32),
                rows[self.emb_static_cols].values.astype(int),
                rows[self.emb_temporal_cols].values.astype(int),
                y.values.astype(int))