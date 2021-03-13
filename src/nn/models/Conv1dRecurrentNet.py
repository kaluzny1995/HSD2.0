import torch
import functools
import operator

from ...constants import LABELS


class Conv1dRecurrentNet(torch.nn.Module):
    def __init__(self, nn_type='recurrent', in_size=100, in_channels=3, out_channels=32, kernel_size=5, n_convs=2, out_size=len(LABELS),
                 input_dim=(3, 256), hidden_size=100, n_layers=1, drop_prob=0.1, bidirectional=False):
        super(Conv1dRecurrentNet, self).__init__()

        self.__in_size = in_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_convs = n_convs - 1
        self.out_size = out_size
        self.input_dim = input_dim

        self.hidden_size = hidden_size if not bidirectional else hidden_size * 2
        self.n_layers = n_layers

        modules = list([])
        modules.append(torch.nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=self.kernel_size, stride=1))
        modules.append(torch.nn.ReLU(inplace=True))
        modules.append(torch.nn.MaxPool1d(kernel_size=2))
        for i in range(self.n_convs):
            modules.append(torch.nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels,
                                           kernel_size=self.kernel_size, stride=1))
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.MaxPool1d(kernel_size=2))

        self.feature_extractor = torch.nn.Sequential(*modules)

        n_rec = self.feature_extractor(torch.rand(1, *self.input_dim)).shape[-1]

        nn_class = torch.nn.RNN if nn_type.find('recurrent') >= 0 else torch.nn.LSTM if nn_type.find('lstm') >= 0 \
            else torch.nn.GRU

        self.rec = nn_class(n_rec, hidden_size, self.n_layers,
                            batch_first=True, dropout=drop_prob, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(self.hidden_size, out_size)

    def forward(self, vector):
        vector = self.feature_extractor(vector)

        out, _ = self.rec(vector)
        out = self.fc(out[:, -1, :])

        return out
