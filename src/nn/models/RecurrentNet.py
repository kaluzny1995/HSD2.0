import torch

from ...constants import LABELS


class RecurrentNet(torch.nn.Module):
    def __init__(self, in_size=256, hidden_size=100, out_size=len(LABELS),
                 n_layers=4, drop_prob=0.1, bidirectional=False):
        super(RecurrentNet, self).__init__()
        self.hidden_size = hidden_size if not bidirectional else hidden_size * 2
        self.n_layers = n_layers

        self.rec = torch.nn.RNN(in_size, hidden_size, self.n_layers,
                                batch_first=True, dropout=drop_prob, bidirectional=bidirectional)
        #self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(self.hidden_size, out_size)

    def forward(self, x):
        out, _ = self.rec(x)
        #out = self.relu(out[:, -1])
        out = self.fc(out[:, -1, :])
        return out


class LSTMNet(torch.nn.Module):
    def __init__(self, in_size=256, hidden_size=100, out_size=len(LABELS),
                 n_layers=4, drop_prob=0.1, bidirectional=False):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size if not bidirectional else hidden_size * 2
        self.n_layers = n_layers

        self.lstm = torch.nn.LSTM(in_size, hidden_size, self.n_layers,
                                  batch_first=True, dropout=drop_prob, bidirectional=bidirectional)
        #self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(self.hidden_size, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # out = self.relu(out[:, -1])
        out = self.fc(out[:, -1, :])
        return out


class GRUNet(torch.nn.Module):
    def __init__(self, in_size=256, hidden_size=100, out_size=len(LABELS),
                 n_layers=4, drop_prob=0.1, bidirectional=False):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size if not bidirectional else hidden_size * 2
        self.n_layers = n_layers

        self.gru = torch.nn.GRU(in_size, hidden_size, self.n_layers,
                                batch_first=True, dropout=drop_prob, bidirectional=bidirectional)
        #self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(self.hidden_size, out_size)

    def forward(self, x):
        out, _ = self.gru(x)
        # out = self.relu(out[:, -1])
        out = self.fc(out[:, -1, :])
        return out
