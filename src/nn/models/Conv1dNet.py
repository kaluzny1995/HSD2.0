import torch
import functools
import operator

from ...constants import LABELS


class Conv1dNet(torch.nn.Module):
    def __init__(self, in_size=100, in_channels=3, out_channels=20, kernel_size=7, n_convs=5, out_size=len(LABELS), input_dim=(3, 256)):
        super(Conv1dNet, self).__init__()
        self.__in_size = in_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_convs = n_convs - 1
        self.out_size = out_size
        self.input_dim = input_dim

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

        n_dense = functools.reduce(operator.mul,
                                   list(self.feature_extractor(torch.rand(1, *self.input_dim)).shape))

        self.classifier = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=n_dense, out_features=100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=100, out_features=out_size),
        )

    def forward(self, vector):
        batch_size = vector.size(0)

        vector = self.feature_extractor(vector)
        vector = vector.view(batch_size, -1)  # flatten the vector
        out = self.classifier(vector)

        return out
