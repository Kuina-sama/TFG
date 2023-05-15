### IMPORTANTE. SI USO TFIDF EL TAMAÑO DE ENTRADA ES EL TAMAÑO DE VOCABULARIO. ASEGURATE DE NO USAR PADDING

from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

import numpy as np

from transformers import get_scheduler

from tqdm.auto import tqdm

import utils_generic as generic
from rrnn_modelo import LSTM_enc

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


class LogisticRegression(nn.Module):
    def __init__(self, sequence_max_len):
        super().__init__()
        self.sequence_max_len = sequence_max_len
        self.linear = nn.Linear(self.sequence_max_len, 1)

    def forward(self, input_ids=None, **kwargs):

        out = self.linear(input_ids.float())
        return out


class LogisticRegressionDep(nn.Module):
    def __init__(self, sequence_max_len, lstm_hidden_dim, dep_vocab):
        super().__init__()
        self.sequence_max_len = sequence_max_len
        self.emb_dim = 20
        self.lstm_hidden_dim = lstm_hidden_dim
        self.LSTM_model = LSTM_enc(
            self.emb_dim, self.lstm_hidden_dim, dep_vocab)
        self.linear = nn.Linear(self.sequence_max_len +
                                self.lstm_hidden_dim*2, 1)

    def forward(self, input_ids=None, dep_tags=None, **kwargs):

        lstm_out, (lstm_hidden_state, cell_state) = self.LSTM_model(dep_tags)
        lstm_cat = torch.cat(
            [lstm_hidden_state[0], lstm_hidden_state[1]], dim=1)

        output = torch.cat((input_ids.float(), lstm_cat), 1)
        out = self.linear(output)

        return out


class MultiTaskLR(nn.Module):
    def __init__(self, sequence_max_len):
        super().__init__()

        self.tasks = ['to', 'as', 'about']
        self.sequence_max_len = sequence_max_len
        self.linear = nn.Linear(self.sequence_max_len, 1)

    def forward(self, input_ids=None, **kwargs):

        tasks_output = {v: self.linear(input_ids.float()) for v in self.tasks}

        return tasks_output


# Comprobar si funciona (?????)
class MultiTaskRRNNDep(nn.Module):
    """Modelo multitask que considera también los tags de dependencia"""

    def __init__(self, sequence_max_len, lstm_hidden_dim, dep_vocab):
        super().__init__()
        self.tasks = ['to', 'as', 'about']
        self.sequence_max_len = sequence_max_len
        self.lstm_hidden_dim = lstm_hidden_dim

        self.LSTM_model = LSTM_enc(
            self.emb_dim, self.lstm_hidden_dim, dep_vocab)
        self.linear = nn.Linear(self.sequence_max_len +
                                self.lstm_hidden_dim*2, 1)

    def forward(self, input_ids=None, dep_tags=None, **kwargs):

        lstm_out, (lstm_hidden_state, cell_state) = self.LSTM_model(dep_tags)
        lstm_cat = torch.cat(
            [lstm_hidden_state[0], lstm_hidden_state[1]], dim=1)

        output = torch.cat((input_ids.float(), lstm_cat), 1)

        tasks_output = {v: self.linear(input_ids.float()) for v in self.tasks}

        return tasks_output
