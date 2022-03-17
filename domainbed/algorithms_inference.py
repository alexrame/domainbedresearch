import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from pyhessian import hessian
except:
    hessian = None
from domainbed import networks, algorithms
from domainbed.lib import misc, diversity_metrics

ALGORITHMS = [
    "ERM",
    "Ensembling",
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]



class ERM(algorithms.ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        algorithms.Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams['nonlinear_classifier']
        )
        self.num_classes = num_classes
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self._init_swa()
        self._init_temperature(init_optimizers=False)

    def _init_from_save_dict(self, save_dict):
        self.load_state_dict(save_dict["model_dict"])
        if self.hparams['swa']:
            self.swa.load_state_dict(save_dict["swa_dict"])


class Ensembling(algorithms.Ensembling):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """
        """
        algorithms.Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)

        featurizers = [
            networks.Featurizer(input_shape, self.hparams) for _ in range(self.hparams["num_members"])
        ]
        classifiers = [
            networks.Classifier(
                featurizers[0].n_outputs,
                self.num_classes,
                self.hparams["nonlinear_classifier"],
                hparams=self.hparams,
            ) for _ in range(self.hparams["num_members"])
        ]
        self.networks = nn.ModuleList(
            [
                nn.Sequential(featurizers[member], classifiers[member])
                for member in range(self.hparams["num_members"])
            ]
        )
        self._init_swa()
        self.soup = misc.Soup(self.networks)
        self._init_temperature(init_optimizers=False)

    def _init_from_save_dict(self, save_dict):
        self.load_state_dict(save_dict["model_dict"])
        self.soup.update()
        if self.hparams['swa']:
            for member in range(self.hparams["num_members"]):
                self.swa[member].load_state_dict(save_dict[f"swa{member}_dict"])
            self.soupswa.update()
