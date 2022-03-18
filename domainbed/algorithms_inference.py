import os
import copy
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
        return ERM
        # raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
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
            if self.hparams['swa'] == 1:
                self.swa.network_swa.load_state_dict(save_dict["swa_dict"])
            else:
                for i in range(self.hparams['swa']):
                    self.swas[i].network_swa.load_state_dict(save_dict[f"swa{i}_dict"])

class GroupDRO(ERM):
    def __init__(self, *args, **kwargs):
        ERM.__init__(self, *args, **kwargs)
        self.register_buffer("q", torch.Tensor([0 for _ in range(self.num_domains)]))


class Fishr(ERM):
    def __init__(self, *args, **kwargs):
        ERM.__init__(self, *args, **kwargs)
        self.register_buffer('update_count', torch.tensor([0]))


class Ensembling(algorithms.Ensembling):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """
        """
        algorithms.Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)

        featurizers = [
            networks.Featurizer(input_shape, self.hparams)
            for _ in range(self.hparams["num_members"])
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
        self._init_temperature(init_optimizers=False)

    def _init_from_save_dict(self, save_dict):
        self.load_state_dict(save_dict["model_dict"])
        self.soup = misc.Soup(self.networks)
        if self.hparams['swa']:
            for member in range(self.hparams["num_members"]):
                self.swas[member].network_swa.load_state_dict(save_dict[f"swa{member}_dict"])
            self.soupswa = misc.Soup(networks=[swa.network_swa for swa in self.swas])


class Soup(algorithms.Ensembling):

    def __init__(self, input_shape, num_classes, num_domains):
        """
        """
        algorithms.Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams={})

        self.networks = []
        self.swas = []
        self.soup = None
        self.soupswa = None
        self._init_memory()

    def to(self, device):
        algorithms.Algorithm.to(self, device)
        for net in self.networks:
            net.to(device)
        if self.soup is not None:
            self.soup.network_soup.to(device)
        for swa in self.swas:
            swa.to(device)
        if self.soupswa is not None:
            self.soupswa.network_soup.to(device)

    def train(self, *args):
        algorithms.Algorithm.train(self, *args)
        for net in self.networks:
            net.train(*args)
        self.soup.network_soup.train(*args)
        for swa in self.swas:
            swa.train(*args)
        self.soupswa.network_soup.train(*args)

    def _init_memory(self):
        self.memory = {"net": 0, "swa": 0}

    def add_new_algorithm(self, algorithm):
        self._init_memory()
        if isinstance(algorithm, ERM):
            self.networks.append(copy.deepcopy(algorithm.network))
            self.memory["net"] += 1
        else:
            assert isinstance(algorithm, Ensembling)
            for member, network in enumerate(algorithm.networks):
                if int(os.environ.get('NETMEMBER', member)) == member:
                    self.networks.append(copy.deepcopy(network))
                    self.memory["net"] += 1

        if algorithm.swa is not None:
            self.memory["swa"] += 1
            self.swas.append(copy.deepcopy(algorithm.swa.network_swa))
        if algorithm.swas is not None:
            for swa in algorithm.swas:
                if int(os.environ.get('SWAMEMBER', member)) == member:
                    self.swas.append(copy.deepcopy(swa.network_swa))
                    self.memory["swa"] += 1

        self.soup = misc.Soup(self.networks)
        self.soupswa = misc.Soup(self.swas)

    def delete_last(self):
        self.networks = self.networks[:-self.memory["net"]]
        self.swas = self.swas[:-self.memory["swa"]]
        self.soup = misc.Soup(self.networks)
        self.soupswa = misc.Soup(self.swas)
        self._init_memory()

    def num_members(self):
        return len(self.networks)

    def predict(self, x):
        results = {}
        batch_logits = []
        batch_logits_swa = []

        for num_member in range(self.num_members()):
            logits = self.networks[num_member](x)
            batch_logits.append(logits)
            results["net" + str(num_member)] = logits
            logits_swa = self.swas[num_member](x)
            batch_logits_swa.append(logits_swa)
            results["swa" + str(num_member)] = logits_swa

        results["net"] = torch.mean(torch.stack(batch_logits, dim=0), 0)
        results["swa"] = torch.mean(torch.stack(batch_logits_swa, dim=0), 0)
        results["soup"] = self.soup.network_soup(x)
        results["soupswa"] = self.soupswa.network_soup(x)

        return results

    def accuracy(self, loader, device, **kwargs):
        self.eval()
        batch_classes = []
        dict_stats = {}
        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x = x.to(device)
                dict_logits = self.predict(x)
                y = y.to(device)
                batch_classes.append(y)
                for key in dict_logits.keys():
                    if key not in dict_stats:
                        dict_stats[key] = {
                            "logits": [],
                            "preds": [],
                            "confs": [],
                            "correct": [],
                            "probs": [],
                        }
                    logits = dict_logits[key]

                    preds = logits.argmax(1)
                    probs = torch.softmax(logits, dim=1)
                    dict_stats[key]["logits"].append(logits.cpu())
                    dict_stats[key]["probs"].append(probs.cpu())
                    dict_stats[key]["preds"].append(preds.cpu())
                    dict_stats[key]["correct"].append(preds.eq(y).float().cpu())
                    dict_stats[key]["confs"].append(probs.max(dim=1)[0].cpu())

        for key0 in dict_stats:
            for key1 in dict_stats[key0]:
                dict_stats[key0][key1] = torch.cat(dict_stats[key0][key1])

        results = {}

        for key in dict_stats:
            results[f"Accuracies/acc_{key}"] = sum(dict_stats[key]["correct"].numpy()
                                                  ) / len(dict_stats[key]["correct"].numpy())
            results[f"Calibration/ece_{key}"] = misc.get_ece(
                dict_stats[key]["confs"].numpy(), dict_stats[key]["correct"].numpy()
            )

        results["Accuracies/acc_netm"] = np.mean(
            [results[f"Accuracies/acc_net{key}"] for key in range(self.num_members())]
        )
        results["Calibration/ece_netm"] = np.mean(
            [results[f"Calibration/ece_net{key}"] for key in range(self.num_members())]
        )
        results["Accuracies/acc_swam"] = np.mean(
            [results[f"Accuracies/acc_swa{key}"] for key in range(self.num_members())]
        )
        results["Calibration/ece_swam"] = np.mean(
            [results[f"Calibration/ece_swa{key}"] for key in range(self.num_members())]
        )
        for key in range(self.num_members()):
            del results[f"Accuracies/acc_net{key}"]
            del results[f"Calibration/ece_net{key}"]
            del results[f"Accuracies/acc_swa{key}"]
            del results[f"Calibration/ece_swa{key}"]

        targets_torch = torch.cat(batch_classes)
        for regex in ["swanet", "swa0swa1", "net01"]:
            if regex == "swanet":
                key0 = "swa"
                key1 = "net"
            elif regex == "swa0net0":
                key0 = "swa0"
                key1 = "net0"
            elif regex == "swa0swa1":
                key0 = "swa0"
                key1 = "swa1"
            elif regex == "net01":
                key0 = "net0"
                key1 = "net1"
            elif regex == "soupnet":
                key0 = "soup"
                key1 = "net"
            elif regex == "soupswaswa":
                key0 = "soupswa"
                key1 = "swa"
            elif regex == "soupswasoup":
                key0 = "soupswa"
                key1 = "soup"
            else:
                raise ValueError(regex)

            if key0 not in dict_stats:
                continue
            if key1 not in dict_stats:
                continue

            targets = targets_torch.cpu().numpy()
            preds0 = dict_stats[key0]["preds"].numpy()
            preds1 = dict_stats[key1]["preds"].numpy()
            results[f"Diversity/{regex}ratio"] = diversity_metrics.ratio_errors(
                targets, preds0, preds1
            )
            results[f"Diversity/{regex}qstat"] = diversity_metrics.Q_statistic(
                targets, preds0, preds1
            )

        return results
