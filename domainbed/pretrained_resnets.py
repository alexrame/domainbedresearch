import torch
import torch.nn as nn
# try:
#     import pytorch_lightning as pl
# except:
#     pass
import torch.nn.functional as F
import torchvision


class PretrainedResnetExtractor(nn.Module):
    def __init__(
        self, name, pretrained=True, freeze=True,
    ):
        """
        In eval mode, batch statistics are not updated.
        """
        super().__init__()
        resnet = torchvision.models.__dict__[name](pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        last_dim = list(resnet.children())[-1].in_features
        self.last_dim = last_dim
        self.n_outputs = last_dim

        self.freeze = freeze
        if self.freeze:
            for p in self.features.parameters():
                p.requires_grad = False
            self.features.eval()

    def forward(self, x):
        if self.freeze:
            for p in self.features.parameters():
                p.requires_grad = False
            self.features.eval()
        bsize = x.shape[0]
        feats = self.features(x).view(bsize, -1)
        return feats


class PretrainedResnet18(PretrainedResnetExtractor):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__("resnet18", pretrained=pretrained, freeze=freeze)


class PretrainedResnet50(PretrainedResnetExtractor):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__("resnet50", pretrained=pretrained, freeze=freeze)
