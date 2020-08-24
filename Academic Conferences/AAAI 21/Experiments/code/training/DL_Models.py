from abc import ABC, abstractmethod
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import r2_score
import sys
import copy
from copy import deepcopy
import math
import time

from . import utils
from .Loaders import MLPDataset, Dataset
from .Base_Models import BaseModel
from .Metrics import BCE

np.random.seed(0)
torch.random.manual_seed(0)

class DLModel(torch.nn.Module, ABC):
    def __init__(self, config, feature_groups = None, dtypes = None, model_name = None, device = None):
        super().__init__()
        self.config = config
        self.model_config = self.config.training.models.get(model_name, {})
        self.param_config = self.model_config.params
        self.seq_len = config.data.forecast_horizon
        self.loss_func = utils.loss_func(config)
        self.usecols, self.dtypes  = utils.get_usecols(feature_groups, dtypes, self.model_config.feature_groups, 
                                        self.model_config.exclude_cols, config)
        self.feature_groups = feature_groups
        self.max_log_y = 1
        self.model_name = model_name
        self.training_state_dict = {'backward': True, 'optimizer_step': True, 'zero_grad': True}
        if self.config.training.GPU:
            if device is None:
                device = ['cpu', 'cuda'][torch.cuda.is_available()]
            self.device = torch.device(device)
        
    @abstractmethod
    def _init_network(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward(self, *args):
        pass

    def network_epoch(self, train_loader, optimizer, scheduler, scheduler_name, parameter_average, regularize, SWA, 
                val_loader = None, epochs = None, early_stopper = None):
        self.reset_training_state()
        optimizer.zero_grad()

        self.scheduler_name = scheduler_name
        self.regularize = regularize
        self.SWA = SWA
        cur_best = None
        current_elements_count = 0
        best_losses = [None] * self.param_config.parameter_averaging.window
        best_states = [None] * self.param_config.parameter_averaging.window

        if self.config.training.GPU:
            self.to(self.device)
        
        for epoch in range(epochs):
            start = time.time()
            self.train_epoch(train_loader, optimizer, scheduler)
            if val_loader:
                loss = self.val_epoch(val_loader)          
            if parameter_average:
                state, self.current_elements_count = utils.parameter_averaging(self, loss, self.param_config, best_losses, best_states, current_elements_count)
            else:
                state, cur_best = utils.parameter_best(self, loss, cur_best)            
            utils.save_best_checkpoint(state)
            sys.stdout.write('\r')
            sys.stdout.write('| Model %s Epoch [%3d/%3d] Val_Error: %.6f Time: %.2f' % (self.model_name, epoch + 1, epochs  , loss, (time.time() - start)))
            if self.scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(loss)
            early_stopper(loss, self)
            if early_stopper.early_stop:
                break
        return self
            

    def train_epoch(self, train_loader, optimizer, scheduler):
        self.train()
        train_loss = 0
        for batch in train_loader:
            if self.config.training.GPU:
                batch = [d.to(self.device) for d in batch]
            batch = [d for d in batch]
            *features, target = batch
            target = target.flatten()
            if self.training_state_dict['zero_grad']:
                optimizer.zero_grad()
            pred = self(*features)
            if self.config.training.GPU:
                loss = self.loss_func(pred, target)
            else:
                loss = self.loss_func(pred, torch.tensor(target , dtype=torch.float32))
            optimizer.zero_grad()
            if self.regularize:
                loss += utils.regularizer(self.param_config, self.parameters())
            if self.training_state_dict['backward']:
                loss.backward()
            if self.training_state_dict['optimizer_step']:
                optimizer.step()
                if self.scheduler_name == 'CyclicLR':
                    scheduler.step()
            train_loss += loss.item()
        if self.SWA:
            optimizer.swap_swa_sgd()


    def val_epoch(self, val_loader):
        self.eval()
        val_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if self.config.training.GPU:
                    batch = [d.to(self.device) for d in batch]
                batch = [d for d in batch]
                *features, target = batch
                target = target.flatten()
                pred = self(*features)
                if self.config.training.GPU:
                    loss = self.loss_func(pred, target)
                else:
                    loss = self.loss_func(pred, torch.tensor(target , dtype=torch.float32))
                val_loss += loss.item()
        return val_loss/len(val_loader)

    def pred_epoch(self, Data_loader):
        self.eval()
        preds = []
        with torch.no_grad():
            for batch in Data_loader:
                if self.config.training.GPU:
                    batch = [d.to(self.device) for d in batch]
                batch = [d for d in batch]
                *features, target = batch
                self.load_state_dict(torch.load(open('../data/training/save_checkpoint/best.pt', 'rb')))
                pred = torch.sigmoid(self(*features))
                preds.append(pred)
        return torch.cat(preds)

    def reset_training_state(self):
        """
        Reset the training states
        Args:
            None
        """

        self.training_state_dict.update({'backward': True, 'optimizer_step': True, 'zero_grad': True})


class MLP(DLModel, BaseModel):

    def _init_network(self, cont_in_features, emb_in_features, out_features=1, emb_out_features=None,
                      compression_factor=None):
        """

        Args:
            cont_in_features:
            emb_in_features:
            out_features:
            emb_out_features:
            compression_factor:

        Returns:

        """
        self.embs = nn.ModuleList()
        if emb_out_features is None:
            emb_out_features = [min(int(np.ceil(math.sqrt(x))), self.param_config.network_params.max_embedding) for x in emb_in_features]
        for in_s, out_s in zip(emb_in_features, emb_out_features):
            self.embs.append(nn.Embedding(in_s, out_s))

        cont_in_features += sum(emb_out_features)
        self.decoder = nn.Linear(cont_in_features , 2**(int(np.math.log(cont_in_features,2))))
        cont_in_features = 2**(int(np.math.log(cont_in_features,2)))
        self.lins = nn.ModuleList()
        i = utils.compression_multiplier(cont_in_features)
        while cont_in_features > i * (compression_factor**2):
            self.lins.append(nn.Linear(cont_in_features, cont_in_features // compression_factor))
            cont_in_features //= compression_factor
        self.out = nn.Linear(cont_in_features, out_features)

    def forward(self, cont_data, emb_data):
        """

        Args:
            cont_data:
            emb_data:

        Returns:

        """
        mlp_data = [cont_data]
        for i, emb in enumerate(self.embs):
            x = emb_data[:, i]
            mlp_data.append(emb(x))

        ip = torch.cat(mlp_data, dim=-1)
        ip = F.relu(self.decoder(ip))
        for lin in self.lins:
            ip = F.relu(lin(ip))
        return (self.out(ip)).flatten()

    def fit(self, X, y, val_data=None, dataloader_params=None, optimizer=None, scheduler=None, parameter_average = None, 
                regularize = None, SWA = None, feature_group_type = None, metrics=None, trial=None, val_round=None, **fit_params):
                
        """

        Args:
            X:
            y:
            val_data:
            dataloader_params:
            optimizer_params:
            scheduler_params:
            metrics:
            trial:
            val_round:
            **fit_params:

        Returns:

        """
        self.feature_group_type = feature_group_type
        self.batch_size = min(int(np.ceil(len(X) / 50)), self.param_config.dataloader_params.batch_size)

        if self.feature_group_type:
            X, val_data = utils.get_loader_data(self.feature_groups, self.feature_group_type, self.model_config, X, val_data = val_data, fit = True)
        if dataloader_params:
            self.param_config.dataloader_params.update(dataloader_params)
        dataloader_params = dict(self.param_config.dataloader_params, batch_size = self.batch_size)

        train_loader = DataLoader(MLPDataset(X, self._scale_y(y, fit=True)), **dataloader_params)
        val_loader = None
        if val_data:
            dataloader_params = dict(self.param_config.dataloader_params, batch_size = self.batch_size)
            val_loader = DataLoader(MLPDataset(val_data[0], self._scale_y(val_data[1])), **dataloader_params)

        cont_in_features = len(train_loader.dataset.cont_cols)
        emb_in_features = X.loc[:, train_loader.dataset.emb_cols].astype(int).max().add(1).tolist()
        self._init_network(cont_in_features, emb_in_features , compression_factor= self.param_config.network_params.compression_factor)
        self.optimizer, scheduler, scheduler_name, early_stopper = utils.DL_model_params(self.parameters(), self.param_config, optimizer, scheduler, SWA, train_loader)
        self.network_epoch(train_loader, self.optimizer, scheduler, scheduler_name, parameter_average, regularize, SWA, val_loader, self.param_config.network_params.epochs, early_stopper)
        return self

    def predict(self, X):
        """

        Args:
            X:

        Returns:

        """

        if self.feature_group_type:
            X = utils.get_loader_data(self.feature_groups, self.feature_group_type, self.model_config, X, val_data = None, fit = False)
        dataloader_params = dict(self.param_config.dataloader_params,batch_size = self.batch_size, shuffle=False)
        data_loader = DataLoader(MLPDataset(X), **dataloader_params)
        preds = self.pred_epoch(data_loader).cpu().detach().numpy()
        return self._invert_y(preds)

class LSTM(DLModel, BaseModel):

    def _init_network(self, cont_temporal_in_features, cont_static_in_features, emb_temporal_in_features,
                      emb_static_in_features, out_features=1, emb_temporal_out_features=None,
                      emb_static_out_features=None, compression_factor=None):
        """

        Args:
            cont_temporal_in_features:
            cont_static_in_features:
            emb_temporal_in_features:
            emb_static_in_features:
            out_features:
            emb_temporal_out_features:
            emb_static_out_features:
            compression_factor:

        Returns:

        """
        self.out_features = out_features
        # temporal embedding
        self.embs_temporal = nn.ModuleList()
        if emb_temporal_out_features is None:
            emb_temporal_out_features = [min(int(np.ceil(math.sqrt(x))), self.param_config.network_params.max_embedding) for x in emb_temporal_in_features]
        for in_s, out_s in zip(emb_temporal_in_features, emb_temporal_out_features):
            self.embs_temporal.append(nn.Embedding(in_s, out_s))

        # Static embedding
        self.embs_static = nn.ModuleList()
        if emb_static_out_features is None:
            emb_static_out_features = [min(int(np.ceil(math.sqrt(x))), self.param_config.network_params.max_embedding) for x in emb_static_in_features]
        for in_s, out_s in zip(emb_static_in_features, emb_static_out_features):
            self.embs_static.append(nn.Embedding(in_s, out_s))

        cont_temporal_in_features += sum(emb_temporal_out_features)
        cont_static_in_features += sum(emb_static_out_features)

        # Initialize the static encoder
        in_enc_stat_dim = cont_static_in_features
        out_enc_stat_dim = 2 ** (int(np.math.log(cont_static_in_features, 2)))
        self.encoder_static = nn.Linear(in_enc_stat_dim, out_enc_stat_dim)

        # Initialize  LSTM #Input: seq_length x batch_size x input_size
        in_rnn_dim = cont_temporal_in_features
        out_rnn_dim = 2 ** (int(np.math.log(cont_temporal_in_features, 2)))
        self.rnn = nn.LSTM(in_rnn_dim, out_rnn_dim, num_layers = self.param_config.LSTM_params.lstm_layers)

        # Initialize  decoder
        cont_in_features = out_enc_stat_dim + out_rnn_dim
        self.decoder = nn.Linear(cont_in_features , 2**(int(np.math.log(cont_in_features,2))))
        cont_in_features = 2**(int(np.math.log(cont_in_features,2)))
        self.lins = nn.ModuleList()
        i = utils.compression_multiplier(cont_in_features)
        while cont_in_features > i * (compression_factor**2):
            self.lins.append(nn.Linear(cont_in_features, cont_in_features // compression_factor))
            cont_in_features //= compression_factor
        self.out = nn.Linear(cont_in_features, out_features)

    def forward(self, cont_static_data, cont_temporal_data, emb_static_data, emb_temporal_data):
        """

        Args:
            cont_static_data:
            cont_temporal_data:
            emb_static_data:
            emb_temporal_data:

        Returns:

        """
        # Get the static data in 2 Dimension
        emb_static_data = emb_static_data.view(emb_static_data.size()[0] * emb_static_data.size()[1],
                                               emb_static_data.size()[2])
        cont_static_data = cont_static_data.view(cont_static_data.size()[0] * cont_static_data.size()[1],
                                                 cont_static_data.size()[2])

        # Temporal data with temporal embedding
        temporal_data = [cont_temporal_data]
        for i, emb in enumerate(self.embs_temporal):
            x = emb_temporal_data[:, :, i]
            temporal_data.append(emb(x))

        # Static data with Static embedding
        static_data = [cont_static_data]
        for i, emb in enumerate(self.embs_static):
            x = emb_static_data[:, i]
            static_data.append(emb(x))

        # MLP Encoder layer
        mlp_enc_data = torch.cat(static_data, dim=1)
        mlp_enc = F.relu(self.encoder_static(mlp_enc_data))

        # LSTM Encoder data  - # Initialize  LSTM #Input: seq_length x batch_size x input_size
        rnn_enc_data = torch.cat(temporal_data, dim=2)
        rnn_enc_data_size = rnn_enc_data.size()
        rnn_enc_data = rnn_enc_data.view(rnn_enc_data_size[1], rnn_enc_data_size[0], -1)
        rnn_out, c_n = self.rnn(rnn_enc_data)
        rnn_sizes = rnn_out.size()
        # flattens rnn_out to batch_size*seq_len,rnn_out_size
        rnn_out = rnn_out.contiguous().view(rnn_sizes[0] * rnn_sizes[1], rnn_sizes[2])

        ip = torch.cat((mlp_enc, rnn_out), dim=1)
        ip = F.relu(self.decoder(ip))
        for lin in self.lins:
            ip = F.relu(lin(ip))
        output = (self.out(ip)).flatten()
        return output

    def fit(self, X, y, val_data=None, dataloader_params=None, optimizer=None, scheduler=None, parameter_average = None, 
                regularize = None, SWA = None, feature_group_type = None, metrics=None, trial=None, val_round=None, **fit_params):
        """

        Args:
            X:
            y:
            val_data:
            dataloader_params:
            optimizer_params:
            scheduler_params:
            metrics:
            trial:
            val_round:
            **fit_params:

        Returns:

        """
        
        self.feature_group_type = feature_group_type
        self.batch_size = min(int(np.ceil(len(X) / 50)), self.param_config.dataloader_params.batch_size)
        if self.feature_group_type:
            X, val_data = utils.get_loader_data(self.feature_groups, self.feature_group_type, self.model_config, X, val_data = val_data, fit = True)
        if dataloader_params:
            self.dataloader_params.update(dataloader_params)
        dataloader_params = dict(self.param_config.dataloader_params, batch_size = self.batch_size)
        train_loader = DataLoader(Dataset(X, self._scale_y(y, fit=True), static_cols= self.config.training.static_cols,
                                          seq_len=self.seq_len, feature_groups = self.feature_groups), **dataloader_params)
        val_loader = None
        if val_data:
            dataloader_params = dict(self.param_config.dataloader_params, batch_size = self.batch_size)
            val_loader = DataLoader(Dataset(val_data[0], self._scale_y(val_data[1]), static_cols= self.config.training.static_cols,
                                            seq_len=self.seq_len, feature_groups = self.feature_groups), **dataloader_params)

        cont_temporal_in_features = len(train_loader.dataset.cont_temporal_cols)
        cont_static_in_features = len(train_loader.dataset.cont_static_cols)
        emb_temporal_in_features = X.loc[:, train_loader.dataset.emb_temporal_cols].astype(int).max().add(1).tolist()
        emb_static_in_features = X.loc[:, train_loader.dataset.emb_static_cols].astype(int).max().add(1).tolist()
        self._init_network(cont_temporal_in_features, cont_static_in_features, emb_temporal_in_features,
                           emb_static_in_features, compression_factor= self.param_config.network_params.compression_factor)
        self.optimizer, scheduler, scheduler_name, early_stopper = utils.DL_model_params(self.parameters(), self.param_config, optimizer, scheduler, SWA, train_loader)
        self.network_epoch(train_loader, self.optimizer, scheduler, scheduler_name, parameter_average, regularize, SWA, val_loader, self.param_config.network_params.epochs, early_stopper)
        return self

    def predict(self, X):
        """

        Args:
            X:

        Returns:

        """

        if self.feature_group_type:
            X = utils.get_loader_data(self.feature_groups, self.feature_group_type, self.model_config, X, val_data = None, fit = False)
        dataloader_params = dict(self.param_config.dataloader_params, batch_size = self.batch_size , shuffle=False)
        data_loader = DataLoader(Dataset(X, static_cols= self.config.training.static_cols,
                                         seq_len= self.seq_len, feature_groups = self.feature_groups), **dataloader_params)
        preds = self.pred_epoch(data_loader).cpu().detach().numpy()
        return self._invert_y(preds)

class WAVENET(DLModel, BaseModel):

    def _init_network(self, cont_temporal_in_features, cont_static_in_features, emb_temporal_in_features,
                      emb_static_in_features, out_features=1, emb_temporal_out_features=None,
                      emb_static_out_features=None, compression_factor=None):
        """

        Args:
            cont_temporal_in_features:
            cont_static_in_features:
            emb_temporal_in_features:
            emb_static_in_features:
            out_features:
            emb_temporal_out_features:
            emb_static_out_features:
            compression_factor:

        Returns:

        """
        
        self.out_features = out_features
        # temporal embedding
        self.embs_temporal = nn.ModuleList()
        if emb_temporal_out_features is None:
            emb_temporal_out_features = [min(int(np.ceil(math.sqrt(x))), self.param_config.network_params.max_embedding) for x in emb_temporal_in_features]
        for in_s, out_s in zip(emb_temporal_in_features, emb_temporal_out_features):
            self.embs_temporal.append(nn.Embedding(in_s, out_s))

        # Static embedding
        self.embs_static = nn.ModuleList()
        if emb_static_out_features is None:
            emb_static_out_features = [min(int(np.ceil(math.sqrt(x))), self.param_config.network_params.max_embedding) for x in emb_static_in_features]
        for in_s, out_s in zip(emb_static_in_features, emb_static_out_features):
            self.embs_static.append(nn.Embedding(in_s, out_s))

        cont_temporal_in_features += sum(emb_temporal_out_features)
        cont_static_in_features += sum(emb_static_out_features)

        # Initialize the static encoder
        in_enc_stat_dim = cont_static_in_features
        out_enc_stat_dim = 2 ** (int(np.math.log(cont_static_in_features, 2)))
        self.encoder_static = nn.Linear(in_enc_stat_dim, out_enc_stat_dim)

        # Initialize  CONV1D #Input: (batch_size * channels * sequence)
        if self.seq_len > 2:
            kernel_size_conv = self.param_config.convnet_params.kernel_size
            dilation_conv = 2
        else:
            kernel_size_conv = 1
            dilation_conv = 1

        in_cnn1_dim = cont_temporal_in_features
        out_cnn1_dim = 2 ** (int(np.math.log(cont_temporal_in_features, 2)))
        out_cnn2_dim = self.seq_len

        #stack of dilated causal convolutions
        #pad = (kernel_size -1) * dilation + 1
        self.causal_conv_lins = nn.ModuleList()
        for dilation in range(self.param_config.convnet_params.dilations):
            dilation = dilation + 1
            kernel_size = min(self.param_config.convnet_params.kernel_size, dilation)
            self.causal_conv_lins.append(torch.nn.Conv1d(in_cnn1_dim, in_cnn1_dim, kernel_size=kernel_size, padding=(kernel_size - 1) * dilation ,dilation=dilation))
             
        self.conv1 = torch.nn.Conv1d(in_cnn1_dim, out_cnn1_dim, kernel_size=kernel_size_conv, dilation=dilation_conv)
        self.conv2 = torch.nn.Conv1d(out_cnn1_dim, out_cnn2_dim, kernel_size=1)
        self.conv_transpose = nn.ConvTranspose1d(out_cnn2_dim, out_cnn2_dim, kernel_size=kernel_size_conv,
                                                 dilation=dilation_conv)

        # Initialize  decoder
        cont_in_features = out_enc_stat_dim + out_cnn2_dim
        self.decoder = nn.Linear(cont_in_features , 2**(int(np.math.log(cont_in_features,2))))
        cont_in_features = 2**(int(np.math.log(cont_in_features,2)))
        self.lins = nn.ModuleList()
        i = utils.compression_multiplier(cont_in_features)
        while cont_in_features > i * (compression_factor**2):
            self.lins.append(nn.Linear(cont_in_features, cont_in_features // compression_factor))
            cont_in_features //= compression_factor
        self.out = nn.Linear(cont_in_features, out_features)
        

    def forward(self, cont_static_data, cont_temporal_data, emb_static_data, emb_temporal_data):
        """

        Args:
            cont_static_data:
            cont_temporal_data:
            emb_static_data:
            emb_temporal_data:

        Returns:

        """
        # Get the static data in 2 Dimension
        emb_static_data = emb_static_data.view(emb_static_data.size()[0] * emb_static_data.size()[1],
                                               emb_static_data.size()[2])
        cont_static_data = cont_static_data.view(cont_static_data.size()[0] * cont_static_data.size()[1],
                                                 cont_static_data.size()[2])

        # Temporal data with temporal embedding
        temporal_data = [cont_temporal_data]
        for i, emb in enumerate(self.embs_temporal):
            x = emb_temporal_data[:, :, i]
            temporal_data.append(emb(x))

        # Static data with Static embedding
        static_data = [cont_static_data]
        for i, emb in enumerate(self.embs_static):
            x = emb_static_data[:, i]
            static_data.append(emb(x))

        # MLP Encoder layer
        mlp_enc_data = torch.cat(static_data, dim=1)
        mlp_enc = F.relu(self.encoder_static(mlp_enc_data))
        # mlp_sizes = mlp_enc.size()

        # CONV1D Encoder data  -  Input: (batch_size * channels * sequence)
        conv1d_enc_data = torch.cat(temporal_data, dim=2)
        conv1d_enc_data_size = conv1d_enc_data.size()
        conv1d_enc_data = conv1d_enc_data.view(-1, conv1d_enc_data_size[2], conv1d_enc_data_size[1])
        
        for causal_conv_lin in self.causal_conv_lins:
            causal_conv = F.relu(causal_conv_lin(conv1d_enc_data))
            causal_conv = causal_conv[:, :, :-causal_conv_lin.padding[0]]

        conv1 = F.relu(self.conv1(causal_conv))
        conv2 = F.relu(self.conv2(conv1))
        conv_transpose = F.relu(self.conv_transpose(conv2))
        conv_transpose = conv_transpose.view(conv_transpose.size()[0] * (conv_transpose.size()[2]),
                                             conv_transpose.size()[1])

        ip = torch.cat((mlp_enc, conv_transpose), dim=1)
        ip = F.relu(self.decoder(ip))
        for lin in self.lins:
            ip = F.relu(lin(ip))
        output = (self.out(ip)).flatten()
        return output

    def fit(self, X, y, val_data=None, dataloader_params=None, optimizer=None, scheduler=None, parameter_average = None, 
                    regularize = None, SWA = None, feature_group_type = None, metrics=None, trial=None, val_round=None, **fit_params):
        """

        Args:
            X:
            y:
            val_data:
            trial:
            val_round:
            dataloader_params:
            optimizer_params:
            scheduler_params:
            metrics:
            **fit_params:

        Returns:

        """

        self.feature_group_type = feature_group_type
        self.batch_size = min(int(np.ceil(len(X) / 50)), self.param_config.dataloader_params.batch_size)
        if self.feature_group_type:
            X, val_data = utils.get_loader_data(self.feature_groups, self.feature_group_type, self.model_config, X, val_data = val_data, fit = True)
        if dataloader_params:
            self.dataloader_params.update(dataloader_params)
        dataloader_params = dict(self.param_config.dataloader_params, batch_size = self.batch_size)
        train_loader = DataLoader(Dataset(X, self._scale_y(y, fit=True), static_cols= self.config.training.static_cols,
                                          seq_len=self.seq_len, feature_groups = self.feature_groups), **dataloader_params)
        val_loader = None
        if val_data:
            dataloader_params = dict(self.param_config.dataloader_params, batch_size = self.batch_size)
            val_loader = DataLoader(Dataset(val_data[0], self._scale_y(val_data[1]), static_cols= self.config.training.static_cols,
                                            seq_len=self.seq_len, feature_groups = self.feature_groups), **dataloader_params)

        cont_temporal_in_features = len(train_loader.dataset.cont_temporal_cols)
        cont_static_in_features = len(train_loader.dataset.cont_static_cols)
        emb_temporal_in_features = X.loc[:, train_loader.dataset.emb_temporal_cols].astype(int).max().add(1).tolist()
        emb_static_in_features = X.loc[:, train_loader.dataset.emb_static_cols].astype(int).max().add(1).tolist()
        self._init_network(cont_temporal_in_features, cont_static_in_features, emb_temporal_in_features,
                           emb_static_in_features, compression_factor= self.param_config.network_params.compression_factor)
        self.optimizer, scheduler, scheduler_name, early_stopper = utils.DL_model_params(self.parameters(), self.param_config, optimizer, scheduler, SWA, train_loader)
        self.network_epoch(train_loader, self.optimizer, scheduler, scheduler_name, parameter_average, regularize, SWA, val_loader, self.param_config.network_params.epochs, early_stopper)
        return self

    def predict(self, X):
        """

        Args:
            X:

        Returns:

        """

        if self.feature_group_type:
            X = utils.get_loader_data(self.feature_groups, self.feature_group_type, self.model_config, X, val_data = None, fit = False)
        dataloader_params = dict(self.param_config.dataloader_params, batch_size = self.batch_size , shuffle=False)
        data_loader = DataLoader(Dataset(X, static_cols= self.config.training.static_cols,
                                         seq_len= self.seq_len, feature_groups = self.feature_groups), **dataloader_params)
        preds = self.pred_epoch(data_loader).cpu().detach().numpy()
        return self._invert_y(preds)
        
class CNN_LSTM(DLModel, BaseModel):

    def _init_network(self, cont_temporal_in_features, cont_static_in_features, emb_temporal_in_features,
                      emb_static_in_features, out_features=1, emb_temporal_out_features=None,
                      emb_static_out_features=None, compression_factor=None):
        """

        Args:
            cont_temporal_in_features:
            cont_static_in_features:
            emb_temporal_in_features:
            emb_static_in_features:
            out_features:
            emb_temporal_out_features:
            emb_static_out_features:
            compression_factor:

        Returns:

        """
        
        self.out_features = out_features
        # temporal embedding
        self.embs_temporal = nn.ModuleList()
        if emb_temporal_out_features is None:
            emb_temporal_out_features = [min(int(np.ceil(math.sqrt(x))), self.param_config.network_params.max_embedding) for x in emb_temporal_in_features]
        for in_s, out_s in zip(emb_temporal_in_features, emb_temporal_out_features):
            self.embs_temporal.append(nn.Embedding(in_s, out_s))

        # Static embedding
        self.embs_static = nn.ModuleList()
        if emb_static_out_features is None:
            emb_static_out_features = [min(int(np.ceil(math.sqrt(x))), self.param_config.network_params.max_embedding) for x in emb_static_in_features]
        for in_s, out_s in zip(emb_static_in_features, emb_static_out_features):
            self.embs_static.append(nn.Embedding(in_s, out_s))

        cont_temporal_in_features += sum(emb_temporal_out_features)
        cont_static_in_features += sum(emb_static_out_features)

        # Initialize the static encoder
        in_enc_stat_dim = cont_static_in_features
        out_enc_stat_dim = 2 ** (int(np.math.log(cont_static_in_features, 2)))
        self.encoder_static = nn.Linear(in_enc_stat_dim, out_enc_stat_dim)

        # Initialize  LSTM #Input: seq_length x batch_size x input_size
        in_rnn_dim = cont_temporal_in_features
        out_rnn_dim = 2 ** (int(np.math.log(cont_temporal_in_features, 2)))
        self.rnn = nn.LSTM(in_rnn_dim, out_rnn_dim, num_layers = self.param_config.LSTM_params.lstm_layers)

        # Initialize  CONV1D #Input: (batch_size * channels * sequence)
        if self.seq_len > 2:
            kernel_size_conv = self.param_config.convnet_params.kernel_size
            dilation_conv = 2
        else:
            kernel_size_conv = 1
            dilation_conv = 1

        in_cnn1_dim = cont_temporal_in_features
        out_cnn1_dim = 2 ** (int(np.math.log(cont_temporal_in_features, 2)))
        out_cnn2_dim = self.seq_len

        #stack of dilated causal convolutions
        #pad = (kernel_size -1) * dilation + 1
        self.causal_conv_lins = nn.ModuleList()
        for dilation in range(self.param_config.convnet_params.dilations):
            dilation = dilation + 1
            kernel_size = min(self.param_config.convnet_params.kernel_size, dilation)
            self.causal_conv_lins.append(torch.nn.Conv1d(in_cnn1_dim, in_cnn1_dim, kernel_size=kernel_size, padding=(kernel_size - 1) * dilation ,dilation=dilation))

        self.conv1 = torch.nn.Conv1d(in_cnn1_dim, out_cnn1_dim, kernel_size=kernel_size_conv, dilation=dilation_conv)
        self.conv2 = torch.nn.Conv1d(out_cnn1_dim, out_cnn2_dim, kernel_size=1)
        self.conv_transpose = nn.ConvTranspose1d(out_cnn2_dim, out_cnn2_dim, kernel_size=kernel_size_conv,
                                                 dilation=dilation_conv)

        # Initialize  decoder
        cont_in_features = out_enc_stat_dim + out_rnn_dim + out_cnn2_dim
        self.decoder = nn.Linear(cont_in_features , 2**(int(np.math.log(cont_in_features,2))))
        cont_in_features = 2**(int(np.math.log(cont_in_features,2)))
        self.lins = nn.ModuleList()
        i = utils.compression_multiplier(cont_in_features)
        while cont_in_features > i * (compression_factor**2):
            self.lins.append(nn.Linear(cont_in_features, cont_in_features // compression_factor))
            cont_in_features //= compression_factor
        self.out = nn.Linear(cont_in_features, out_features)

    def forward(self, cont_static_data, cont_temporal_data, emb_static_data, emb_temporal_data):
        """

        Args:
            cont_static_data:
            cont_temporal_data:
            emb_static_data:
            emb_temporal_data:

        Returns:

        """
        # Get the static data in 2 Dimension
        emb_static_data = emb_static_data.view(emb_static_data.size()[0] * emb_static_data.size()[1],
                                               emb_static_data.size()[2])
        cont_static_data = cont_static_data.view(cont_static_data.size()[0] * cont_static_data.size()[1],
                                                 cont_static_data.size()[2])

        # Temporal data with temporal embedding
        temporal_data = [cont_temporal_data]
        for i, emb in enumerate(self.embs_temporal):
            x = emb_temporal_data[:, :, i]
            temporal_data.append(emb(x))

        # Static data with Static embedding
        static_data = [cont_static_data]
        for i, emb in enumerate(self.embs_static):
            x = emb_static_data[:, i]
            static_data.append(emb(x))

        # MLP Encoder layer
        mlp_enc_data = torch.cat(static_data, dim=1)
        mlp_enc = F.relu(self.encoder_static(mlp_enc_data))
        # mlp_sizes = mlp_enc.size()

        # LSTM Encoder data  - # Initialize  LSTM #Input: seq_length x batch_size x input_size
        rnn_enc_data = torch.cat(temporal_data, dim=2)
        rnn_enc_data_size = rnn_enc_data.size()
        rnn_enc_data = rnn_enc_data.view(rnn_enc_data_size[1], rnn_enc_data_size[0], -1)
        rnn_out, c_n = self.rnn(rnn_enc_data)
        rnn_sizes = rnn_out.size()
        # flattens rnn_out to batch_size*seq_len,rnn_out_size
        rnn_out = rnn_out.contiguous().view(rnn_sizes[0] * rnn_sizes[1], rnn_sizes[2])

        # CONV1D Encoder data  -  Input: (batch_size * channels * sequence)
        conv1d_enc_data = torch.cat(temporal_data, dim=2)
        conv1d_enc_data_size = conv1d_enc_data.size()
        conv1d_enc_data = conv1d_enc_data.view(-1, conv1d_enc_data_size[2], conv1d_enc_data_size[1])

        for causal_conv_lin in self.causal_conv_lins:
            causal_conv = F.relu(causal_conv_lin(conv1d_enc_data))
            causal_conv = causal_conv[:, :, :-causal_conv_lin.padding[0]]
        
        conv1 = F.relu(self.conv1(causal_conv))
        conv2 = F.relu(self.conv2(conv1))
        conv_transpose = F.relu(self.conv_transpose(conv2))
        conv_transpose = conv_transpose.view(conv_transpose.size()[0] * (conv_transpose.size()[2]),
                                             conv_transpose.size()[1])

        ip = torch.cat((mlp_enc, rnn_out, conv_transpose), dim=1)
        ip = F.relu(self.decoder(ip))
        for lin in self.lins:
            ip = F.relu(lin(ip))
        output = (self.out(ip)).flatten()
        return output

    def fit(self, X, y, val_data=None, dataloader_params=None, optimizer=None, scheduler=None, parameter_average = None, 
                regularize = None, SWA = None, feature_group_type = None, metrics=None, trial=None, val_round=None, **fit_params):
        """

        Args:
            X:
            y:
            val_data:
            trial:
            val_round:
            dataloader_params:
            optimizer_params:
            scheduler_params:
            metrics:
            **fit_params:

        Returns:

        """
        
        self.feature_group_type = feature_group_type
        self.batch_size = min(int(np.ceil(len(X) / 50)), self.param_config.dataloader_params.batch_size)
        if self.feature_group_type:
            X, val_data = utils.get_loader_data(self.feature_groups, self.feature_group_type, self.model_config, X, val_data = val_data, fit = True)
        if dataloader_params:
            self.dataloader_params.update(dataloader_params)
        dataloader_params = dict(self.param_config.dataloader_params, batch_size = self.batch_size)
        train_loader = DataLoader(Dataset(X, self._scale_y(y, fit=True), static_cols= self.config.training.static_cols,
                                          seq_len=self.seq_len, feature_groups = self.feature_groups), **dataloader_params)
        val_loader = None
        if val_data:
            dataloader_params = dict(self.param_config.dataloader_params, batch_size = self.batch_size)
            val_loader = DataLoader(Dataset(val_data[0], self._scale_y(val_data[1]), static_cols= self.config.training.static_cols,
                                            seq_len=self.seq_len, feature_groups = self.feature_groups), **dataloader_params)

        cont_temporal_in_features = len(train_loader.dataset.cont_temporal_cols)
        cont_static_in_features = len(train_loader.dataset.cont_static_cols)
        emb_temporal_in_features = X.loc[:, train_loader.dataset.emb_temporal_cols].astype(int).max().add(1).tolist()
        emb_static_in_features = X.loc[:, train_loader.dataset.emb_static_cols].astype(int).max().add(1).tolist()
        self._init_network(cont_temporal_in_features, cont_static_in_features, emb_temporal_in_features,
                           emb_static_in_features, compression_factor= self.param_config.network_params.compression_factor)
        self.optimizer, scheduler, scheduler_name, early_stopper = utils.DL_model_params(self.parameters(), self.param_config, optimizer, scheduler, SWA, train_loader)
        self.network_epoch(train_loader, self.optimizer, scheduler, scheduler_name, parameter_average, regularize, SWA, val_loader, self.param_config.network_params.epochs, early_stopper)
        return self

    def predict(self, X):
        """

        Args:
            X:

        Returns:

        """

        if self.feature_group_type:
            X = utils.get_loader_data(self.feature_groups, self.feature_group_type, self.model_config, X, val_data = None, fit = False)
        dataloader_params = dict(self.param_config.dataloader_params, batch_size = self.batch_size , shuffle=False)
        data_loader = DataLoader(Dataset(X, static_cols= self.config.training.static_cols,
                                         seq_len= self.seq_len, feature_groups = self.feature_groups), **dataloader_params)
        preds = self.pred_epoch(data_loader).cpu().detach().numpy()
        return self._invert_y(preds)