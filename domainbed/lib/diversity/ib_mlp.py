import torch
import torch.nn as nn
from torch.nn import init

leakyrelu = nn.LeakyReLU(negative_slope=2e-1)


class DiscMLP(nn.Module):

    def __init__(
        self,
        input_size,
        hparams
    ):
        super(DiscMLP, self).__init__()

        self._input_size = input_size
        self._hidden_size = hparams["hidden_size"]
        self._num_hidden_layers = hparams["num_hidden_layers"]
        self._num_members = hparams["num_members"]

        self.fc_1 = nn.Linear(
            self._input_size * self._num_members,
            self._hidden_size, bias=True)
        if self._num_hidden_layers > 1:
            self.fc_2 = nn.Linear(self._hidden_size, self._hidden_size)
        if self._num_hidden_layers > 2:
            self.fc_3 = nn.Linear(self._hidden_size, self._hidden_size)
        self.fc_end = nn.Linear(self._hidden_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    def forward(self, features):
        x = leakyrelu(self.fc_1(features))
        if self._num_hidden_layers > 1:
            x = leakyrelu(self.fc_2(x))
        if self._num_hidden_layers > 2:
            x = leakyrelu(self.fc_3(x))
        logits = self.fc_end(x)
        return logits.reshape(-1, 1)


class CondDiscMLP(nn.Module):

    def __init__(self, input_size, num_classes, hparams):
        super(CondDiscMLP, self).__init__()

        self._input_size = input_size
        self._hidden_size = hparams["hidden_size"]
        self._num_members = hparams["num_members"]
        self._num_hidden_layers = hparams["num_hidden_layers"]

        self.num_classes = num_classes

        self._embedding_layer = nn.Embedding(num_classes, self._input_size)

        self.fc_1 = nn.Linear(
            self._input_size * (self._num_members + 1),
            self._hidden_size, bias=True
        )
        if self._num_hidden_layers > 1:
            self.fc_2 = nn.Linear(self._hidden_size, self._hidden_size)
        if self._num_hidden_layers > 2:
            self.fc_3 = nn.Linear(self._hidden_size, self._hidden_size)
        self.fc_end = nn.Linear(self._hidden_size, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    def forward(self, features, classes):
        embedding_classes = self._embedding_layer(classes)
        x = torch.cat([features, embedding_classes], axis=1)
        x = leakyrelu(self.fc_1(x))
        # x = torch.cat([x, embedding_classes_1], axis=1)
        if self._num_hidden_layers > 1:
            x = leakyrelu(self.fc_2(x))
        if self._num_hidden_layers > 2:
            x = leakyrelu(self.fc_3(x))
        logits = self.fc_end(x)
        logits = self._select_logits(logits, classes)
        return logits.reshape(-1, 1)

    def _select_logits(self, logits, classes):
        classes = classes.reshape(-1)
        selected_logit = logits[torch.arange(logits.size(0)), classes]
        return selected_logit
