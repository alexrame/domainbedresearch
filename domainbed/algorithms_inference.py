import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed import networks, algorithms
from domainbed.lib import misc, diversity_metrics
from domainbed.lib.diversity_metrics import CudaCKA

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

        self.featurizers = [
            networks.Featurizer(input_shape, self.hparams)
            for _ in range(self.hparams["num_members"])
        ]
        classifiers = [
            networks.Classifier(
                self.featurizers[0].n_outputs,
                self.num_classes,
                self.hparams["nonlinear_classifier"],
                hparams=self.hparams,
            ) for _ in range(self.hparams["num_members"])
        ]
        self.networks = nn.ModuleList(
            [
                nn.Sequential(self.featurizers[member], classifiers[member])
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

    def __init__(self, input_shape, num_classes, num_domains, t_scaled="", regexes=None, do_ens=[]):
        """
        """
        algorithms.Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams={})

        self.networks = []
        self._t_scaled = t_scaled
        self.swas = []

        if self._t_scaled:
            self._t_swas = []
            self._t_networks = []

        self.do_ens = do_ens

        self.create_soups()
        self._init_memory()
        self.regexes = [] if regexes is None else regexes

    def get_temperature(self, key, return_optim=False):
        assert not return_optim
        assert self._t_scaled

        if key in ["net", "soup"]:
            return torch.mean(torch.stack(self._t_networks)).view(-1)
        if key in ["swa", "soupswa"]:
            return torch.mean(torch.stack(self._t_swas)).view(-1)

        i = key[-1]
        if key == "net" + str(i):
            return self._t_networks[int(i)]

        if key == "swa" + str(i):
            return self._t_swas[int(i)]
        return None

    def create_soups(self):
        self.soup = misc.Soup(self.networks)
        self.soupswa = misc.Soup(self.swas)
        # if self._t_scaled == "swa":
        #     self.soup.update_tscaled(self._t_networks)
        #     self.soupswa.update_tscaled(self._t_swas)

    def to(self, device):
        algorithms.Algorithm.to(self, device)
        if self.soup is not None:
            self.soup.network_soup.to(device)
        if self.soupswa is not None:
            self.soupswa.network_soup.to(device)

        if "net" in self.do_ens:
            for net in self.networks:
                net.to(device)
            if self._t_scaled:
                self._t_networks = [t.to(device) for t in self._t_networks]

        if "swa" in self.do_ens:
            for swa in self.swas:
                swa.to(device)
            if self._t_scaled:
                self._t_swas = [t.to(device) for t in self._t_swas]

    def train(self, *args):
        algorithms.Algorithm.train(self, *args)
        self.soup.network_soup.train(*args)
        self.soupswa.network_soup.train(*args)
        if "net" in self.do_ens:
            for net in self.networks:
                net.train(*args)
        if "swa" in self.do_ens:
            for swa in self.swas:
                swa.train(*args)

    def _init_memory(self):
        self.memory = {"net": 0, "swa": 0}

    def add_new_algorithm(self, algorithm):
        self._init_memory()
        if isinstance(algorithm, ERM):
            self.networks.append(copy.deepcopy(algorithm.network))
            if self._t_scaled:
                self._t_networks.append(algorithm.get_temperature("net"))
            self.memory["net"] += 1
        else:
            assert isinstance(algorithm, Ensembling)
            for member, network in enumerate(algorithm.networks):
                if int(os.environ.get('NETMEMBER', member)) == member:
                    self.networks.append(copy.deepcopy(network))
                    self.memory["net"] += 1
                    if self._t_scaled:
                        self._t_networks.append(algorithm.get_temperature("net" + str(member)))

        if algorithm.swa is not None:
            self.memory["swa"] += 1
            self.swas.append(copy.deepcopy(algorithm.swa.network_swa))
            if self._t_scaled:
                self._t_swas.append(algorithm.get_temperature("swa"))

        if algorithm.swas is not None:
            for member, swa in enumerate(algorithm.swas):
                if str(os.environ['SWAMEMBER']) == str(member):
                    self.swas.append(copy.deepcopy(swa.network_swa))
                    self.memory["swa"] += 1
                    if self._t_scaled:
                        self._t_swas.append(algorithm.get_temperature("swa" + str(member)))
        self.create_soups()

    def delete_last(self):
        self.networks = self.networks[:-self.memory["net"]]
        self.swas = self.swas[:-self.memory["swa"]]
        if self._t_scaled:
            self._t_networks = self._t_networks[:-self.memory["net"]]
            self._t_swas = self._t_swas[:-self.memory["swa"]]

        self.create_soups()
        self._init_memory()

    def num_members(self):
        return len(self.networks)

    def predict(self, x):
        results = {}

        results["soup"] = self.soup.network_soup(x)
        results["soupswa"] = self.soupswa.network_soup(x)

        # results["ens"] = torch.mean(torch.stack([results["soup"], results["soupswa"]], dim=0), 0)
        if not self.do_ens:
            return results

        batch_logits = []
        batch_logits_swa = []

        if self._t_scaled == "full":
            batch_logits_tscaled = []
            batch_logits_swa_tscaled = []

        for num_member in range(self.num_members()):
            if "net" in self.do_ens:
                logits = self.networks[num_member](x)
                batch_logits.append(logits)
                results["net" + str(num_member)] = logits
            if "swa" in self.do_ens:
                logits_swa = self.swas[num_member](x)
                batch_logits_swa.append(logits_swa)
                results["swa" + str(num_member)] = logits_swa
                if self._t_scaled == "full":
                    batch_logits_swa_tscaled.append(logits_swa / self._t_swas[num_member])

        if "net" in self.do_ens:
            results["net"] = torch.mean(torch.stack(batch_logits, dim=0), 0)
            if self._t_scaled == "full":
                results["netts"] = torch.mean(torch.stack(batch_logits_tscaled, dim=0), 0)

        if "swa" in self.do_ens:
            results["swa"] = torch.mean(torch.stack(batch_logits_swa, dim=0), 0)
            if self._t_scaled == "full":
                results["swats"] = torch.mean(torch.stack(batch_logits_swa_tscaled, dim=0), 0)

        return results

    def predict_feat(self, x):
        results = {}

        keys = [key for regex in self.regexes for key in regex.split("_")]
        regexed_nets = [int(key[3:]) for key in keys if key.startswith("net")]
        regexed_swas = [int(key[3:]) for key in keys if key.startswith("swa")]
        # Do this stupid thing because memory error otherwise
        for num_member in range(self.num_members()):
            if num_member in regexed_nets:
                assert "net" in self.do_ens
                results["net" + str(num_member)] = misc.get_featurizer(self.networks[num_member])(x)
            if num_member in regexed_swas:
                assert "swa" in self.do_ens
                results["swa" + str(num_member)] = misc.get_featurizer(self.swas[num_member])(x)

        if "soup" in keys:
            results["soup"] = self.soup.get_featurizer()(x)
        if "soupswa" in keys:
            results["soupswa"] = self.soupswa.get_featurizer()(x)
        return results

    def accuracy(self, loader, device, compute_trace, **kwargs):

        do_calibration = self._t_scaled
        self.eval()
        dict_stats, batch_classes = self.get_dict_stats(
            loader, device, compute_trace, do_temperature=do_calibration, max_feats=10
        )

        results = {}
        for key in dict_stats:
            results[f"Accuracies/acc_{key}"] = sum(dict_stats[key]["correct"].numpy()
                                                  ) / len(dict_stats[key]["correct"].numpy())
            if do_calibration:
                results[f"Calibration/ece_{key}"] = misc.get_ece(
                    dict_stats[key]["confs"].numpy(), dict_stats[key]["correct"].numpy()
                )
                if "confstemp" in dict_stats[key]:
                    results[f"Calibration/ecetemp_{key}"] = misc.get_ece(
                        dict_stats[key]["confstemp"].numpy(), dict_stats[key]["correct"].numpy()
                    )

        if "net" in self.do_ens:
            results["Accuracies/acc_netm"] = np.mean(
                [results[f"Accuracies/acc_net{key}"] for key in range(self.num_members())]
            )
            if do_calibration:
                results["Calibration/ece_netm"] = np.mean(
                    [results[f"Calibration/ece_net{key}"] for key in range(self.num_members())]
                )
            for key in range(self.num_members()):
                del results[f"Accuracies/acc_net{key}"]
                if do_calibration:
                    del results[f"Calibration/ece_net{key}"]

        if "swa" in self.do_ens:
            results["Accuracies/acc_swam"] = np.mean(
                [results[f"Accuracies/acc_swa{key}"] for key in range(self.num_members())]
            )
            if do_calibration:
                results["Calibration/ece_swam"] = np.mean(
                    [results[f"Calibration/ece_swa{key}"] for key in range(self.num_members())]
                )

            for key in range(self.num_members()):
                del results[f"Accuracies/acc_swa{key}"]
                if do_calibration:
                    del results[f"Calibration/ece_swa{key}"]

        targets = torch.cat(batch_classes).cpu().numpy()

        results_div = {}
        count = 0
        for regex in self.regexes:
            if "_" in regex:
                key0 = regex.split("_")[0]
                key1 = regex.split("_")[1]
            else:
                raise ValueError(regex)

            if key0 not in dict_stats or key1 not in dict_stats:
                # print(f"{regex} not found for diversity")
                continue
            # regex = regex.replace("soup", "s").replace("swa", "w").replace("net", "n").replace("_", "")
            _results_div = self._compute_diversity(
                targets, dict_stats, key0, key1, compute_trace, device
            )
            count += 1
            for key, value in _results_div.items():
                results_div[key] = results_div.get(key, 0) + value

        for key, value in results_div.items():
            results[key + "_" + str(count)] = value / count

        return results

    def _compute_diversity(self, targets, dict_stats, key0, key1, compute_trace, device):
        results = {}
        preds0 = dict_stats[key0]["preds"].numpy()
        preds1 = dict_stats[key1]["preds"].numpy()
        results[f"Diversity/dr"] = diversity_metrics.ratio_errors(targets, preds0, preds1)
        # results[f"Diversity/{regex}qstat"] = diversity_metrics.Q_statistic(
        #     targets, preds0, preds1
        # )
        if compute_trace and "feats" in dict_stats[key0] and "feats" in dict_stats[key1]:
            feats0 = dict_stats[key0]["feats"]
            feats1 = dict_stats[key1]["feats"]
            results[f"Diversity/df"] = 1. - CudaCKA(device).linear_CKA(feats0, feats1).item()
        return results

    def compute_hessian(self, loader):
        # del self.swas[1:]
        # del self.networks[1:]

        # print(f"Begin Hessian soup")
        results = {}
        results["Flatness/souphess"] = misc.compute_hessian(
            self.soup.network_soup, loader, maxIter=10
        )
        # del self.soup.network_soup

        # # print(f"Begin Hessian soupswa")
        # # results["Flatness/soupswahess"] = misc.compute_hessian(
        # #     self.soupswa.network_soup, loader, maxIter=10
        # # )
        # del self.soupswa.network_soup
        # if "net" in self.do_ens:
        #     print("Begin Hessian net0")
        #     results[f"Flatness/net0hess"] = misc.compute_hessian(
        #         self.networks[0], loader, maxIter=10
        #     )
        #     del self.networks[0]

        # if "swa" in self.do_ens:
        #     print("Begin Hessian swa0")
        #     results[f"Flatness/swa0hess"] = misc.compute_hessian(self.swas[0], loader, maxIter=10)
        #     del self.swas[0]

        return results
