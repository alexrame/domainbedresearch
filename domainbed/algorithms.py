# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch.autograd as autograd
import pdb
import os
import random

from domainbed.lib.diversity_metrics import CudaCKA

from domainbed import networks
from domainbed.lib import misc, diversity_metrics, diversity, sam, losses
from domainbed.lib.misc import count_param, set_requires_grad
import copy
try:
    from torchmetrics import Precision, Recall
except:
    Precision, Recall = None, None
from backpack import backpack, extend
from backpack.extensions import BatchGrad


ALGORITHMS = [
    "ERM",
    "SWA",
    "Ensembling",
    "Fishr",
    "CORAL",
    "GroupDRO",
    'Mixup'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hparams = hparams
        self.num_domains = num_domains

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def get_tb_dict(self):
        return {}



class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams['nonlinear_classifier']
        )
        self.num_classes = num_classes
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self._init_network()
        self._init_swa()
        self._init_optimizer()
        self._init_temperature()

    def _init_network(self):
        if not self.hparams["shared_init"]:
            return
        path = str(self.hparams["shared_init"]) + "_" + str(self.num_classes)

        if os.environ.get("CREATE_INIT"):
            assert not os.path.exists(path)
            print("Not loading because creating init with linear probing")
        else:
            assert os.path.exists(path)
            weights = torch.load(path)
            self.network.load_state_dict(weights)

    def _save_network_for_future(self):
        path = str(self.hparams["shared_init"]) + "_" + str(self.num_classes)
        if os.environ.get("CREATE_INIT"):
            assert not os.path.exists(path)
            print('Saving network weights for future')
            torch.save(self.network.state_dict(), path)

    def _init_temperature(self, init_optimizers=True):
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=True)
        if init_optimizers:
            self.t_optimizer = torch.optim.Adam(
                [self.temperature],
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )
        if self.hparams['swa']:
            if self.swa is not None or self.hparams.get('num_members'):
                self.swa_temperature = nn.Parameter(torch.ones(1), requires_grad=True)
                if init_optimizers:
                    self.t_swa_optimizer = torch.optim.Adam(
                        [self.swa_temperature],
                        lr=self.hparams["lr"],
                        weight_decay=self.hparams["weight_decay"],
                    )
            if self.swas is not None:
                num_temps = len(self.swas)
                self.swa_temperatures = []
                if init_optimizers:
                    self.t_swa_optimizers = []
                for _ in range(num_temps):
                    self.swa_temperatures.append(nn.Parameter(torch.ones(1), requires_grad=True))
                    if init_optimizers:
                        self.t_swa_optimizers.append(
                            torch.optim.Adam(
                                [self.swa_temperatures[-1]],
                                lr=self.hparams["lr"],
                                weight_decay=self.hparams["weight_decay"],
                            )
                        )
                if self.hparams.get('num_members'):
                    self.soupswa_temperature = nn.Parameter(torch.ones(1), requires_grad=True)
                    if init_optimizers:
                        self.t_soupswa_optimizer = torch.optim.Adam(
                            [self.soupswa_temperature],
                            lr=self.hparams["lr"],
                            weight_decay=self.hparams["weight_decay"],
                        )
        if self.hparams.get('num_members'):
            self.soup_temperature = nn.Parameter(torch.ones(1), requires_grad=True)
            if init_optimizers:
                self.t_soup_optimizer = torch.optim.Adam(
                    [self.soup_temperature],
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams["weight_decay"],
                )
            self.net_temperatures = []
            if init_optimizers:
                self.t_net_optimizers = []
            for _ in range(self.hparams.get('num_members')):
                self.net_temperatures.append(nn.Parameter(torch.ones(1), requires_grad=True))
                if init_optimizers:
                    self.t_net_optimizers.append(
                        torch.optim.Adam(
                            [self.net_temperatures[-1]],
                            lr=self.hparams["lr"],
                            weight_decay=self.hparams["weight_decay"],
                        )
                    )

    def get_temperature(self, key, return_optim=False):
        if key == "net":
            if return_optim:
                return self.temperature, self.t_optimizer
            return self.temperature
        if key == "swa":
            if self.swa is None:
                if return_optim:
                    return None, None
                return None
            if return_optim:
                return self.swa_temperature, self.t_swa_optimizer
            return self.swa_temperature

        i = key[-1]
        if key == "swa" + str(i) and self.swas is not None:
            if return_optim:
                return self.swa_temperatures[int(i)], self.t_swa_optimizers[int(i)]
            return self.swa_temperatures[int(i)]

        if not self.hparams.get("num_members"):
            if return_optim:
                return None, None
            return None

        if key == "net" + str(i):
            if return_optim:
                return self.net_temperatures[int(i)], self.t_net_optimizers[int(i)]
            return self.net_temperatures[int(i)]

        if key == "soup":
            if return_optim:
                return self.soup_temperature, self.t_soup_optimizer
            return self.soup_temperature
        if key == "soupswa":
            if return_optim:
                return self.soupswa_temperature, self.t_soupswa_optimizer
            return self.soupswa_temperature

        if return_optim:
            return None, None
        return None

    def _init_optimizer(self):
        if os.environ.get("CREATE_INIT"):
            parameters_to_be_optimized = self.classifier.parameters()
        else:
            parameters_to_be_optimized = self.network.parameters()
        self.optimizer = torch.optim.Adam(
                parameters_to_be_optimized,
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

    def _init_swa(self):
        self.swa = None
        self.swas = None
        if self.hparams['swa']:
            if self.hparams['swa'] == 1:
                self.swa = misc.SWA(self.network, hparams=self.hparams)
            else:
                _start = [100, 2500, 4000, 4500, 4800]
                self.swas = [
                    misc.SWA(self.network, hparams=self.hparams, swa_start_iter=_start[i])
                    for i in range(self.hparams['swa'])
                ]

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_classes = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.network(all_x), all_classes)

        output_dict = {'loss': loss}

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_swa()

        return {key: value.item() for key, value in output_dict.items()}

    def update_swa(self):
        if self.swa is not None:
            self.swa.update()
        if self.swas is not None:
            for swa in self.swas:
                swa.update()

    def predict(self, x):
        results = {"net": self.network(x)}

        if self.swa is not None:
            results["swa"] = self.swa.network_swa(x)
        if self.swas is not None:
            for i, swa in enumerate(self.swas):
                results[f"swa{i}"] = swa.network_swa(x)

        return results

    def predict_feat(self, x):
        feats_network = self.featurizer(x)
        if self.hparams['swa']:
            feats_swa = self.swa.get_featurizer()(x)
            results = {"swa": feats_swa, "net": feats_network}
        else:
            results = {"net": feats_network}
        return results

    def to(self, device):
        Algorithm.to(self, device)
        if self.swa is not None:
            self.swa.network_swa.to(device)
        if self.swas is not None:
            for swa in self.swas:
                swa.network_swa.to(device)

    def train(self, *args):
        Algorithm.train(self, *args)
        if self.swa is not None:
            self.swa.network_swa.train(*args)
        if self.swas is not None:
            for swa in self.swas:
                swa.network_swa.train(*args)

    def get_dict_stats(self, loader, device, compute_trace, do_temperature=True, max_feats=float("inf"), list_temperatures=None):
        batch_classes = []
        dict_stats = {}
        if list_temperatures is None:
            list_temperatures = ["net", "net0", "net1", "swa", "swa0", "swa1", "soup", "soupswa"]
        with torch.no_grad():
            for i, batch in enumerate(loader):
                x, y = batch
                x = x.to(device)
                dict_logits = self.predict(x)
                if compute_trace and i < max_feats:
                    dict_feats = self.predict_feat(x)
                else:
                    dict_feats = {}
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
                    if key in dict_feats:
                        if "feats" not in dict_stats[key]:
                            dict_stats[key]["feats"] = []
                        dict_stats[key]["feats"].append(dict_feats[key])

                    if do_temperature and key in list_temperatures:
                        temperature = self.get_temperature(key)
                        if temperature is None:
                            continue
                        temperature = temperature.to(device)
                        probstemp = torch.softmax(
                            misc.apply_temperature_on_logits(logits.detach(), temperature), dim=1
                        )
                        if "confstemp" not in dict_stats[key]:
                            dict_stats[key]["confstemp"] = []
                        dict_stats[key]["confstemp"].append(probstemp.max(dim=1)[0].cpu())

        for key0 in dict_stats:
            for key1 in dict_stats[key0]:
                dict_stats[key0][key1] = torch.cat(dict_stats[key0][key1])
        return dict_stats, batch_classes

    def accuracy(self, loader, device, compute_trace=False, update_temperature=False, output_temperature=False):
        self.eval()

        dict_stats, batch_classes = self.get_dict_stats(loader, device, compute_trace)

        results = {}
        for key in dict_stats:
            results[f"Accuracies/acc_{key}"] = sum(
                dict_stats[key]["correct"].numpy()) / len(dict_stats[key]["correct"].numpy())
            results[f"Calibration/ece_{key}"] = misc.get_ece(
                dict_stats[key]["confs"].numpy(), dict_stats[key]["correct"].numpy()
            )
            if "confstemp" in dict_stats[key]:
                results[f"Calibration/ecetemp_{key}"] = misc.get_ece(
                    dict_stats[key]["confstemp"].numpy(), dict_stats[key]["correct"].numpy()
                )

        if self.hparams.get("num_members"):
            results["Accuracies/acc_netm"] = np.mean([results[f"Accuracies/acc_net{key}"] for key in range(self.hparams.get("num_members"))])
            results["Calibration/ece_netm"] = np.mean(
                [results[f"Calibration/ece_net{key}"] for key in range(self.hparams.get("num_members"))]
            )
            if self.hparams.get("swa"):
                results["Accuracies/acc_swam"] = np.mean([results[f"Accuracies/acc_swa{key}"] for key in range(self.hparams.get("num_members"))])
                results["Calibration/ece_swam"] = np.mean(
                    [results[f"Calibration/ece_swa{key}"] for key in range(self.hparams.get("num_members"))]
                )

        targets_torch = torch.cat(batch_classes)


        results.update(self._compute_diversity(dict_stats, targets_torch.cpu().numpy(), compute_trace, device))
        results_temp = self._update_temperature_with_stats(
            dict_stats, device, targets_torch, update_temperature=update_temperature
        )
        if output_temperature:
            results.update(results_temp)

        del targets_torch
        self.train()
        return results

    def _compute_diversity(self, dict_stats, targets, compute_trace, device):
        results = {}
        for regex in ["swanet", "swa0net0", "swa0swa1", "net01", "soupnet",]:
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

            preds0 = dict_stats[key0]["preds"].numpy()
            preds1 = dict_stats[key1]["preds"].numpy()
            results[f"Diversity/{regex}ratio"] = diversity_metrics.ratio_errors(
                targets, preds0, preds1
            )
            # results[f"Diversity/{regex}agre"] = diversity_metrics.agreement_measure(
            #     targets, preds0, preds1
            # )
            # results[f"Diversity/{regex}doublefault"] = diversity_metrics.double_fault(targets, preds0, preds1)
            # results[f"Diversity/{regex}singlefault"] = diversity_metrics.single_fault(targets, preds0, preds1)
            results[f"Diversity/{regex}qstat"] = diversity_metrics.Q_statistic(targets, preds0, preds1)
            # del preds0, preds1

            # new div metrics
            if compute_trace and "feats" in dict_stats[key0] and "feats" in dict_stats[key1]:
                feats0 = dict_stats[key0]["feats"]
                feats1 = dict_stats[key1]["feats"]
                results[f"Diversity/{regex}ckac"] = 1. - CudaCKA(device).linear_CKA(feats0, feats1).item()
                # del feats0, feats1

            # probs0 = dict_stats[key0]["probs"].numpy()
            # probs1 = dict_stats[key1]["probs"].numpy()
            # results[f"Diversity/{regex}l2"] = diversity_metrics.l2(probs0, probs1)
            # results[f"Diversity/{regex}nd"] = diversity_metrics.normalized_disagreement(targets, probs0, probs1)
            # del dict_stats[key0], dict_stats[key1]

        return results

    def _update_temperature_with_stats(self, dict_stats, device, targets_torch, update_temperature=True, list_temperatures=None):
        if list_temperatures is None:
            list_temperatures = ["net", "net0", "net1", "swa", "swa0", "swa1", "soup", "soupswa"]
        results = {}
        for key in list_temperatures:
            if key not in dict_stats:
                continue
            if update_temperature:
                temperature, optimizer = self.get_temperature(key, return_optim=True)
            else:
                temperature = self.get_temperature(key, return_optim=False)
            if temperature is None:
                continue

            if update_temperature:
                num_steps_temp = int(os.environ.get("NUMSTEPSTEMP", 20))
                for _ in range(num_steps_temp):
                    logits = dict_stats[key]["logits"].to(device)
                    temperature = temperature.to(device)
                    assert temperature.requires_grad

                    loss_T = F.cross_entropy(
                        misc.apply_temperature_on_logits(logits, temperature), targets_torch
                    )
                    optimizer.zero_grad()
                    loss_T.backward()
                    optimizer.step()
            results["temp_" + key] = temperature.item()
        return results

    def compute_hessian(self, loader):
        # Flatness metrics
        self.eval()
        results = {}
        results[f"Flatness/nettrace"] = misc.compute_hessian(self.network, loader)
        if self.swa is not None:
            results[f"Flatness/swatrace"] = misc.compute_hessian(
                self.swa.network_swa, loader)
        if self.swas is not None:
            results[f"Flatness/swa0trace"] = misc.compute_hessian(self.swas[0].network_swa, loader)
        self.train()
        return results


class SWA(ERM):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        ERM.__init__(self, input_shape, num_classes, num_domains, hparams)
        # diversifier
        self.features_size = self.featurizer.n_outputs
        self.register_buffer('update_count', torch.tensor([0]))

        if self.hparams["diversity_loss"] in [None, "none"]:
            self.member_diversifier = None
        else:
            self.member_diversifier = diversity.DICT_NAME_TO_DIVERSIFIER[
                self.hparams["diversity_loss"]]
            if not isinstance(self.member_diversifier, str):
                self.member_diversifier = self.member_diversifier(
                    hparams=self.hparams,
                    features_size=self.features_size,
                    num_classes=self.num_classes,
                    num_domains=num_domains,
                    num_members=2
                )
    def to(self, device):
        ERM.to(self, device)
        if self.member_diversifier is not None:
            self.member_diversifier.to(device)
            self.member_diversifier.q = self.member_diversifier.q.to(device)

    def update(self, minibatches, unlabeled=None):

        all_x = torch.cat([x for x, y in minibatches])
        all_classes = torch.cat([y for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_loss = F.cross_entropy(all_logits, all_classes, reduction="none")

        output_dict = {'lossmean': all_loss.mean()}

        objective = all_loss.mean()
        if self.member_diversifier is not None:
            raise ValueError()
            # if self.hparams["diversity_loss"] == "sampling":
            #     assert self.hparams.get("div_eta") != 0
            # penalty_active = self.update_count >= self.hparams["penalty_anneal_iters"]
            # if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
            #         # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
            #     # gradient magnitudes that happens at this step.
            #     self._init_optimizer()
            # self.update_count += 1
            # if self.hparams["div_data"] in ["uda", "udaandiid"]:
            #     assert unlabeled is not None
            #     div_objective, div_dict = self.increase_diversity_unlabeled(unlabeled)
            #     if self.hparams["div_data"] in ["udaandiid"]:
            #         output_dict.update({key + "uda": value for key, value in div_dict.items()})
            #     else:
            #         output_dict.update(div_dict)
            # if self.hparams["div_data"] in ["", "none", "udaandiid"]:
            #     bsize = minibatches[0][0].size(0)
            #     div_objective, div_dict = self.increase_diversity_labeled(
            #         all_x, all_features, all_classes, all_loss, all_logits, bsize
            #     )
            #     output_dict.update(div_dict)
            # if penalty_active:
            #     objective = objective + div_objective

        if self.hparams.get('sam'):
            raise ValueError
            # self.optimizer.zero_grad()
            # # first forward-backward pass
            # objective.backward()
            # self.optimizer.first_step(zero_grad=True)
            # # second forward-backward pass
            # objective2 = F.cross_entropy(
            #     self.classifier(self.featurizer(all_x)), all_classes, reduction="mean"
            # )
            # # make sure to do a full forward pass
            # objective2.backward()
            # self.optimizer.second_step()
            # output_dict["objective2"] = objective2
        else:
            self.optimizer.zero_grad()
            objective.backward()
            self.optimizer.step()

        self.update_swa()

        return {key: value.detach().item() for key, value in output_dict.items()}

    def increase_diversity_unlabeled(self, unlabeled):
        bsize = unlabeled[0].size(0)
        num_domains = len(unlabeled)
        all_x = torch.cat(unlabeled)
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        with torch.no_grad():
            swa_features = self.swa.get_featurizer()(all_x)
            swa_logits = self.swa.get_classifier()(all_features)
        assert self.member_diversifier.diversity_type != "sampling"
        kwargs = {
            "features_per_member":
            torch.stack([all_features, swa_features],
                        dim=0).reshape(2, num_domains, bsize, self.features_size),
            "logits_per_member":
            torch.stack([all_logits, swa_logits],
                        dim=0).reshape(2, num_domains, bsize, self.num_classes),
            "classes":
            None,
            "nlls_per_member":
            None
        }
        dict_diversity = self.member_diversifier.forward(**kwargs)

        objective = dict_diversity["loss_div"] * self.hparams["lambda_diversity_loss"]
        if self.hparams["lambda_entropy"]:
            objective = objective + self.hparams["lambda_entropy"
                                                ] * losses.entropy_regularizer(all_logits)
        return objective, dict_diversity

    def increase_diversity_labeled(
        self, all_x, all_features, all_classes, all_loss, all_logits, bsize
    ):
        with torch.no_grad():
            swa_features = self.swa.get_featurizer()(all_x)
            swa_logits = self.swa.get_classifier()(all_features)
            swa_loss = F.cross_entropy(swa_logits, all_classes, reduction="none")
        output_dict = {}
        if self.member_diversifier.diversity_type == "sampling":
            loss_weighted = self.member_diversifier.compute_weighted_loss(
                active_loss=all_loss, sampling_loss=swa_loss
            )
            output_dict["lossw"] = loss_weighted
            objective = loss_weighted - all_loss.mean()
        else:
            kwargs = {
                "features_per_member":
                torch.stack([all_features, swa_features],
                            dim=0).reshape(2, self.num_domains, bsize, self.features_size),
                "logits_per_member":
                torch.stack([all_logits, swa_logits],
                            dim=0).reshape(2, self.num_domains, bsize, self.num_classes),
                "classes":
                all_classes,
                "nlls_per_member":
                torch.stack([all_loss, swa_loss], dim=0).reshape(2, self.num_domains, bsize),
                # "classifiers": [self.classifier, self.swa.get_classifier()]
            }
            dict_diversity = self.member_diversifier.forward(**kwargs)
            output_dict.update(dict_diversity)
            objective = dict_diversity["loss_div"] * self.hparams["lambda_diversity_loss"]

        if self.hparams["lambda_entropy"]:
            objective = objective + self.hparams["lambda_entropy"
                                                ] * losses.entropy_regularizer(all_logits)
        return objective, output_dict


class Ensembling(Algorithm):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """
        """
        Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
        self.loss = nn.CrossEntropyLoss(reduction="mean")

        self.featurizers = [
            networks.Featurizer(input_shape, self.hparams) for _ in range(self.hparams["num_members"])
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
        self._init_network()
        self._init_optimizer()
        self._init_swa()
        self.soup = misc.Soup(self.networks)
        self._init_temperature()

    def _init_network(self):
        if not self.hparams["shared_init"]:
            return
        path = str(self.hparams["shared_init"]) + "_" + str(self.num_classes)
        if os.environ.get("CREATE_INIT"):
            raise ValueError("Can not create init for ensembling")
        else:
            assert os.path.exists(path)
            weights = torch.load(path)
            for i in range(0, self.hparams["num_members"]):
                self.networks[i].load_state_dict(weights)

    def _init_optimizer(self):
        lrs = [self.hparams["lr"] for _ in range(self.hparams["num_members"])]
        if self.hparams.get("lr_ratio", 0) != 0:
            for member in range(self.hparams["num_members"]):
                lrs[member] /= float(self.hparams.get("lr_ratio"))**member
        print("Ensembling lrs: ", lrs)
        self.optimizers = [
            torch.optim.Adam(
                list(self.networks[member].parameters()),
                lr=lrs[member],
                weight_decay=self.hparams["weight_decay"],
            ) for member in range(self.hparams["num_members"])
        ]

    def _init_swa(self):
        self.swa = None
        self.swas = None
        if self.hparams['swa']:
            assert self.hparams['swa'] == 1
            self.swas = [
                misc.SWA(self.networks[member], hparams=self.hparams)
                for member in range(self.hparams["num_members"])
            ]
            self.soupswa = misc.Soup(
                networks=[swa.network_swa for swa in self.swas])


    def update(self, minibatches, unlabeled=None):
        if self.hparams['specialized'] == 1:
            nlls_per_member = self._update_specialized(minibatches)
        elif self.hparams['specialized']:
            # todo create diversity randomly per batch
            nlls_per_member = self._update_partial(minibatches, step=self.hparams['specialized'])
        else:
            nlls_per_member = self._update_full(minibatches)

        out = {"nll": torch.stack(nlls_per_member, dim=0).mean()}
        for key in range(self.hparams["num_members"]):
            out[f"nll_{key}"] = nlls_per_member[key]

        self.soup.update()
        if self.hparams['swa']:
            for i, swa in enumerate(self.swas):
                swa_dict = swa.update()
                # out.update({key + str(i): value for key, value in swa_dict.items()})
            self.soupswa.update()
        return {key: value.item() for key, value in out.items()}

    def _update_full(self, minibatches):
        all_x = torch.cat([x for x, y in minibatches])
        all_classes = torch.cat([y for x, y in minibatches])
        nlls_per_member = []  # (num_classifiers, num_minibatches)

        for member in range(self.hparams["num_members"]):
            logits_member = self.networks[member](all_x)
            nll_member = F.cross_entropy(logits_member, all_classes, reduction="mean")
            nlls_per_member.append(nll_member)

            # optimization step
            self.optimizers[member].zero_grad()
            nll_member.backward()
            self.optimizers[member].step()

        return nlls_per_member

    def _update_partial(self, minibatches, step=2):
        num_domains_per_member = self.hparams["num_members"] // len(minibatches)
        index_per_member = [
            [num_domains_per_member * i + j
             for j in range(num_domains_per_member)]
            for i in range(self.hparams["num_members"])
        ]

        nlls_per_member = []  # (num_members, num_domains_per_member)

        for member in range(self.hparams["num_members"]):
            x_for_member = torch.cat(
                [minibatches[index][0] for index in index_per_member[member]] +
                [minibatches[index][0][member::step] for index in range(len(minibatches)) if index not in index_per_member[member]]
                )
            classes_for_member = torch.cat(
                [minibatches[index][1] for index in index_per_member[member]] + [
                    minibatches[index][1][member::step]
                    for index in range(len(minibatches))
                    if index not in index_per_member[member]
                ]
            )
            logits_member = self.networks[member](x_for_member)
            nll_member = F.cross_entropy(
                logits_member, classes_for_member, reduction="mean"
            )
            nlls_per_member.append(nll_member)

            # optimization step
            self.optimizers[member].zero_grad()
            nll_member.backward()
            self.optimizers[member].step()

        return nlls_per_member

    def _update_specialized(self, minibatches):
        assert self.hparams["num_members"] % len(minibatches) == 0
        num_domains_per_member = self.hparams["num_members"] // len(minibatches)
        index_per_member = [
            [num_domains_per_member * i + j
             for j in range(num_domains_per_member)]
            for i in range(self.hparams["num_members"])
        ]
        x_per_member = [
            torch.cat([minibatches[index][0]
                       for index in index_per_member[i]])
            for i in range(self.hparams["num_members"])
        ]
        classes_per_member = [
            torch.cat([minibatches[index][1]
                       for index in index_per_member[i]])
            for i in range(self.hparams["num_members"])
        ]

        nlls_per_member = []  # (num_members, num_domains_per_member)

        for member in range(self.hparams["num_members"]):
            logits_member = self.networks[member](x_per_member[member])
            nll_member = F.cross_entropy(
                logits_member, classes_per_member[member], reduction="mean"
            )
            nlls_per_member.append(nll_member)

            # optimization step
            self.optimizers[member].zero_grad()
            nll_member.backward()
            self.optimizers[member].step()

        return nlls_per_member

    def to(self, device):
        Algorithm.to(self, device)
        self.soup.network_soup.to(device)
        if self.swas is not None:
            self.soupswa.network_soup.to(device)
            for swa in self.swas:
                swa.network_swa.to(device)

    def train(self, *args):
        Algorithm.train(self, *args)
        self.soup.network_soup.train(*args)
        if self.swas is not None:
            self.soupswa.network_soup.train(*args)
            for swa in self.swas:
                swa.network_swa.train(*args)

    def compute_hessian(self, loader):
        # Flatness metrics
        self.eval()
        results = {}
        results[f"Flatness/souptrace"] = misc.compute_hessian(self.soup.network_soup, loader)
        if self.swas is not None:
            results[f"Flatness/swa0trace"] = misc.compute_hessian(
                self.swas[0].network_swa, loader)
        results[f"Flatness/net0trace"] = misc.compute_hessian(
            self.networks[0], loader)
        self.train()
        return results

    def predict(self, x):
        results = {}
        batch_logits = []
        batch_logits_swa = []

        for num_member in range(self.hparams["num_members"]):
            logits = self.networks[num_member](x)
            batch_logits.append(logits)
            results["net" + str(num_member)] = logits
            if self.hparams['swa']:
                logits_swa = self.swas[num_member].network_swa(x)
                batch_logits_swa.append(logits_swa)
                results["swa" + str(num_member)] = logits_swa
        results["net"] = torch.mean(torch.stack(batch_logits, dim=0), 0)
        if self.hparams['swa']:
            results["swa"] = torch.mean(torch.stack(batch_logits_swa, dim=0), 0)
        results["soup"] = self.soup.network_soup(x)
        results["soupswa"] = self.soupswa.network_soup(x)
        return results

    def predict_feat(self, x):
        results = {}
        for num_member in range(self.hparams["num_members"]):
            if num_member != 0:
                # Do this because memory error otherwise
                continue
            feats = self.featurizers[num_member](x)
            results["net" + str(num_member)] = feats
            if self.hparams['swa']:
                feats_swa = self.swas[num_member].get_featurizer()(x)
                results["swa" + str(num_member)] = feats_swa
        return results

    def accuracy(self, *args, **kwargs):
        return ERM.accuracy(self, *args, **kwargs)

    def get_dict_stats(self, *args, **kwargs):
        return ERM.get_dict_stats(self, *args, **kwargs)

    def _init_temperature(self, *args, **kwargs):
        return ERM._init_temperature(self, *args, **kwargs)

    def get_temperature(self, *args, **kwargs):
        return ERM.get_temperature(self, *args, **kwargs)


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.network(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_swa()

        return {'loss': loss.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in misc.random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.network(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        self.update_swa()

        return {'loss': objective.item()}

class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.network(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_swa()

        return {'loss': loss.item()}


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_lambda'] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        self.update_swa()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)


class Fishr(ERM):
    "Invariant Gradients variances for Out-of-distribution Generalization"

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert backpack is not None, "Install backpack with: 'pip install backpack-for-pytorch==1.3.0'"
        Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = extend(
            networks.Classifier(
                self.featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier'],
            )
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.register_buffer('update_count', torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            misc.MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self._init_network()
        self._init_swa()
        self._init_optimizer()
        self._init_temperature()

    def update(self, minibatches, unlabeled=False):
        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        len_minibatches = [x.shape[0] for x, y in minibatches]

        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.hparams["penalty_anneal_iters"]:
            penalty_weight = self.hparams["lambda"]
            if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        self.update_swa()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += misc.l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains
