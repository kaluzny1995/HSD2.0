import torch

from ...constants import LABELS


class DenseNet(torch.nn.Module):
    def __init__(self, in_size=768, hidden_size=500, out_size=len(LABELS), drop_coeff=0.1, n_linear=5, loaded=False):
        super(DenseNet, self).__init__()
        self._in = in_size
        self._hidden = hidden_size
        self._out = out_size
        self._drop_coeff = drop_coeff
        self._n_linear = n_linear
        self._loaded = loaded

        self.drop = torch.nn.Dropout(self._drop_coeff)
        self.lin_in = torch.nn.Linear(self._in, self._hidden)
        self.relu_in = torch.nn.ReLU(inplace=True)
        self.lin_hidden = torch.nn.Linear(self._hidden, self._hidden)
        self.relu_hidden = torch.nn.ReLU(inplace=True)
        self.lin_out = torch.nn.Linear(self._hidden, self._out)

    def forward(self, vector):
        vector = self.drop(vector)
        vector = self.relu_in(self.lin_in(vector))
        for i in range(self._n_linear):
            vector = self.relu_hidden(self.lin_hidden(vector))
        outputs = self.lin_out(vector)

        return outputs
