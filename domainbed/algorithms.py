# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import pdb
import os
import random
try:
    from pyhessian import hessian
except:
    hessian is None
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
    "IRM",
    "SWA",
    "Ensembling",
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

    CUSTOM_FORWARD = False

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
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self._init_swa()
        self._init_optimizer()
        self._init_temperature()

    def _init_temperature(self):
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=True)
        self.t_optimizer = torch.optim.Adam(
            [self.temperature],
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        if self.hparams['swa']:
            self.swa_temperature = nn.Parameter(torch.ones(1), requires_grad=True)
            self.t_swa_optimizer = torch.optim.Adam(
                [self.swa_temperature],
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )
            if self.hparams.get('num_members', 1) != 1:
                self.swa_temperatures = []
                self.t_swa_optimizers = []
                for _ in range(self.hparams.get('num_members', 1)):
                    self.swa_temperatures.append(nn.Parameter(torch.ones(1), requires_grad=True))
                    self.t_swa_optimizers.append(
                        torch.optim.Adam(
                            [self.swa_temperatures[-1]],
                            lr=self.hparams["lr"],
                            weight_decay=self.hparams["weight_decay"],
                        )
                    )
                self.soupswa_temperature = nn.Parameter(torch.ones(1), requires_grad=True)
                self.t_soupswa_optimizer = torch.optim.Adam(
                    [self.soupswa_temperature],
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams["weight_decay"],
                )
        if self.hparams.get('num_members', 1) != 1:
            self.soup_temperature = nn.Parameter(torch.ones(1), requires_grad=True)
            self.t_soup_optimizer = torch.optim.Adam(
                [self.soup_temperature],
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )
            self.net_temperatures = []
            self.t_net_optimizers = []
            for _ in range(self.hparams.get('num_members', 1)):
                self.net_temperatures.append(nn.Parameter(torch.ones(1), requires_grad=True))
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
            if return_optim:
                return self.swa_temperature, self.t_swa_optimizer
            return self.swa_temperature
        if not self.hparams.get("num_members"):
            if return_optim:
                return None, None
            return None

        if key == "soup":
            if return_optim:
                return self.soup_temperature, self.t_soup_optimizer
            return self.soup_temperature
        if key == "soupswa":
            if return_optim:
                return self.soupswa_temperature, self.t_soupswa_optimizer
            return self.soupswa_temperature
        i = int(key[-1])
        if key == "swa" + str(i):
            if return_optim:
                return self.swa_temperatures[i], self.t_swa_optimizers[i]
            return self.swa_temperatures[i]
        if key == "net" + str(i):
            if return_optim:
                return self.net_temperatures[i], self.t_net_optimizers[i]
            return self.net_temperatures[i]

        raise ValueError(key)
    def _init_swa(self):
        if self.hparams['swa']:
            self.swa = misc.SWA(self.network, hparams=self.hparams)
        else:
            self.swa = None

    def _init_optimizer(self):
        if self.hparams.get('sam'):
            phosam = 10 * self.hparams["phosam"] if self.hparams["samadapt"
                                                                ] else self.hparams["phosam"]
            if self.hparams.get('sam') == "inv":
                phosam = -phosam
            if self.hparams.get('sam') == "onlyswa":
                raise ValueError(self.hparams.get('sam'))
                # self.optimizer = samswa.SAMswa(
                #     self.network.parameters(),
                #     params_swa=self.swa.network_swa.parameters(),
                #     base_optimizer=torch.optim.Adam,
                #     adaptive=self.hparams["samadapt"],
                #     rho=phosam,
                #     lr=self.hparams["lr"],
                #     weight_decay=self.hparams["weight_decay"],
                # )
            else:
                self.optimizer = sam.SAM(
                    self.network.parameters(),
                    torch.optim.Adam,
                    adaptive=self.hparams["samadapt"],
                    rho=phosam,
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams["weight_decay"],
                )
        else:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_classes = torch.cat([y for x, y in minibatches])
        if self.hparams['sam'] == "perdomain":
            domain = random.randint(0, self.num_domains - 1)
            loss = F.cross_entropy(self.network(minibatches[domain][0]), minibatches[domain][1])
        elif self.hparams['sam'] == "swa":
            predictions = self.network(all_x)
            with torch.no_grad():
                predictions_swa = self.swa.network_swa(all_x)
            loss = F.cross_entropy((predictions + predictions_swa) / 2, all_classes)
        elif self.hparams['sam'] == "swasam":
            predictions = self.network(all_x)
            with torch.no_grad():
                predictions_swa = self.swa.network_swa(all_x)
            loss = (1 + self.hparams['swasamcoeff']) * F.cross_entropy(
                (predictions + self.hparams['swasamcoeff'] * predictions_swa) /
                (1 + self.hparams['swasamcoeff']), all_classes
            )
        elif self.hparams['sam'] == "onlyswa":
            loss = F.cross_entropy(self.swa.network_swa(all_x), all_classes)
        else:
            loss = F.cross_entropy(self.network(all_x), all_classes)

        output_dict = {'loss': loss}
        if self.hparams['sam']:
            self.optimizer.zero_grad()
            # first forward-backward pass
            loss.backward()
            self.optimizer.first_step(zero_grad=True)
            # second forward-backward pass
            loss_second_step = F.cross_entropy(self.network(all_x), all_classes)
            # make sure to do a full forward pass
            loss_second_step.backward()
            self.optimizer.second_step()
            output_dict["loss_secondstep"] = loss_second_step
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.hparams['swa']:
            self.swa.update()

        return {key: value.item() for key, value in output_dict.items()}

    def predict(self, x):
        results = {"net": self.network(x)}
        if self.hparams['swa']:
            results["swa"] = self.swa.network_swa(x)
        return results

    # def predict_feat(self, x):
    #     feats_network = self.featurizer(x)
    #     if self.hparams['swa']:
    #         feats_swa = self.swa.get_featurizer()(x)
    #         results = {"swa": feats_swa, "net": feats_network}
    #     else:
    #         results = {"net": feats_network}
    #     return results
    def to(self, device):
        Algorithm.to(self, device)

    def eval(self):
        Algorithm.eval(self)
        if self.hparams['swa']:
            self.swa.network_swa.eval()

    def train(self, *args):
        Algorithm.train(self, *args)
        if self.hparams['swa']:
            self.swa.network_swa.train(*args)

    def accuracy(self, loader, device, compute_trace=False, update_temperature=False):
        self.eval()

        batch_classes = []
        dict_stats = {}
        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x = x.to(device)
                dict_logits = self.predict(x)
                # dict_feats = self.predict_feat(x)
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
                            # "confstemp": []
                        }
                    logits = dict_logits[key]

                    preds = logits.argmax(1)
                    probs = torch.softmax(logits, dim=1)
                    dict_stats[key]["logits"].append(logits.cpu())
                    dict_stats[key]["probs"].append(probs.cpu())
                    dict_stats[key]["preds"].append(preds.cpu())
                    dict_stats[key]["correct"].append(preds.eq(y).float().cpu())
                    dict_stats[key]["confs"].append(probs.max(dim=1)[0].cpu())

                    if key in ["net", "net0", "net1", "swa", "swa0", "swa1", "soup", "soupswa"]:
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
            results["Accuracies/acc_netm"] = np.mean([results[f"Accuracies/acc_net{key}"] for key in self.hparams.get("num_members")])
            results["Calibration/ece_netm"] = np.mean(
                [results[f"Calibration/ece_net{key}"] for key in self.hparams.get("num_members")]
            )
            if self.hparams.get("swa"):
                results["Accuracies/acc_swam"] = np.mean([results[f"Accuracies/acc_swa{key}"] for key in self.hparams.get("num_members")])
                results["Calibration/ece_swam"] = np.mean(
                    [results[f"Calibration/ece_swa{key}"] for key in self.hparams.get("num_members")]
                )

        targets_torch = torch.cat(batch_classes)
        for regex in ["swanet", "swa0net0", "swa0swa1", "net01", "soupnet", "soupswaswa", "soupswasoup"]:
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
            # results[f"Diversity/{regex}agre"] = diversity_metrics.agreement_measure(
            #     targets, preds0, preds1
            # )
            # results[f"Diversity/{regex}doublefault"] = diversity_metrics.double_fault(targets, preds0, preds1)
            # results[f"Diversity/{regex}singlefault"] = diversity_metrics.single_fault(targets, preds0, preds1)
            results[f"Diversity/{regex}qstat"] = diversity_metrics.Q_statistic(
                targets, preds0, preds1
            )

            # # new div metrics
            # probs0 = dict_stats[key0]["probs"].numpy()
            # probs1 = dict_stats[key1]["probs"].numpy()
            # results[f"Diversity/{regex}l2"] = diversity_metrics.l2(probs0, probs1)
            # results[f"Diversity/{regex}nd"] = diversity_metrics.normalized_disagreement(
            #     targets, probs0, probs1
            # )

        # Flatness metrics
        if compute_trace and hessian is not None:
            # feats0 = dict_stats[key0]["feats"]
            # hessian_comp_swa = hessian(
            #     self.swa.get_classifier(), nn.CrossEntropyLoss(reduction='sum'), data=(feats0, targets_torch), cuda=True)

            if not self.hparams.get("num_members"):
                if self.hparams['swa']:
                    hessian_comp_swa = hessian(
                        self.swa.network_swa,
                        nn.CrossEntropyLoss(reduction='mean'),
                        dataloader=loader,
                        cuda=True
                    )
                    results[f"Flatness/swatrace"] = np.mean(hessian_comp_swa.trace())
                hessian_comp_net = hessian(
                    self.network, nn.CrossEntropyLoss(reduction='mean'), dataloader=loader, cuda=True
                )
                results[f"Flatness/nettrace"] = np.mean(hessian_comp_net.trace())
            else:
                hessian_comp_soup = hessian(
                    self.soup.network_soup,
                    nn.CrossEntropyLoss(reduction='mean'),
                    dataloader=loader,
                    cuda=True
                )
                results[f"Flatness/souptrace"] = np.mean(hessian_comp_soup.trace())
                if self.hparams['swa']:
                    hessian_comp_swa = hessian(
                        self.swas[0].network_swa,
                        nn.CrossEntropyLoss(reduction='mean'),
                        dataloader=loader,
                        cuda=True
                    )
                    results[f"Flatness/swa0trace"] = np.mean(hessian_comp_swa.trace())
                hessian_comp_net = hessian(
                    self.networks[0], nn.CrossEntropyLoss(reduction='mean'), dataloader=loader, cuda=True
                )
                results[f"Flatness/net0trace"] = np.mean(hessian_comp_net.trace())

        if update_temperature:
            for _ in range(20):
                for key in ["net", "net0", "net1", "swa", "swa0", "swa1", "soup", "soupswa"]:
                    if key not in dict_stats:
                        continue
                    logits = dict_stats[key]["logits"].to(device)
                    temperature, optimizer = self.get_temperature(key, return_optim=True)
                    if temperature is None:
                        continue
                    temperature = temperature.to(device)
                    assert temperature.requires_grad

                    loss_T = F.cross_entropy(
                        misc.apply_temperature_on_logits(logits, temperature), targets_torch
                    )
                    optimizer.zero_grad()
                    loss_T.backward()
                    optimizer.step()

                results["temp/" + key] = temperature.item()

        self.train()
        return results


class SWA(ERM):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        ERM.__init__(self, input_shape, num_classes, num_domains, hparams)
        # diversifier
        self.features_size = self.featurizer.n_outputs
        self.register_buffer("update_count", torch.tensor([0]))

        self.hparams["num_members"] = 2
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
                    num_domains=num_domains
                )

    def update(self, minibatches, unlabeled=None):
        bsize = minibatches[0][0].size(0)

        all_x = torch.cat([x for x, y in minibatches])
        all_classes = torch.cat([y for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_loss = F.cross_entropy(all_logits, all_classes, reduction="none")

        output_dict = {'lossmean': all_loss.mean()}
        penalty_active = self.update_count >= self.hparams["penalty_anneal_iters"]
        if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
            # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
            # gradient magnitudes that happens at this step.
            self._init_optimizer()
        self.update_count += 1

        if self.hparams["diversity_loss"] == "sampling":
            assert self.hparams.get("div_eta") != 0

        objective = all_loss.mean()
        if self.member_diversifier is not None:
            if self.hparams["div_data"] in ["uda", "udaandiid"]:
                assert unlabeled is not None
                div_objective, div_dict = self.increase_diversity_unlabeled(unlabeled)
                if self.hparams["div_data"] in ["udaandiid"]:
                    output_dict.update({key + "uda": value for key, value in div_dict.items()})
                else:
                    output_dict.update(div_dict)
            if self.hparams["div_data"] in ["", "none", "udaandiid"]:
                div_objective, div_dict = self.increase_diversity_labeled(
                    all_x, all_features, all_classes, all_loss, all_logits, bsize
                )
                output_dict.update(div_dict)
            if penalty_active:
                objective = objective + div_objective

        else:
            pass

        if self.hparams.get('sam'):
            self.optimizer.zero_grad()
            # first forward-backward pass
            objective.backward()
            self.optimizer.first_step(zero_grad=True)
            # second forward-backward pass
            objective2 = F.cross_entropy(
                self.classifier(self.featurizer(all_x)), all_classes, reduction="mean"
            )
            # make sure to do a full forward pass
            objective2.backward()
            self.optimizer.second_step()
            output_dict["objective2"] = objective2
        else:
            self.optimizer.zero_grad()
            objective.backward()
            self.optimizer.step()

        if self.hparams['swa']:
            if self.hparams['swa'] == 1:
                swa_dict = self.swa.update()
                output_dict.update(swa_dict)
            else:
                for i, swa in enumerate(self.swas):
                    swa_dict = swa.update()
                    output_dict.update({key + str(i): value for key, value in swa_dict.items()})

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
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.hparams = hparams
        self.num_members = self.hparams["num_members"]
        featurizers = [
            networks.Featurizer(input_shape, self.hparams) for _ in range(self.num_members)
        ]
        classifiers = [
            networks.Classifier(
                featurizers[0].n_outputs,
                self.num_classes,
                self.hparams["nonlinear_classifier"],
                hparams=self.hparams,
            ) for _ in range(self.num_members)
        ]
        self.networks = nn.ModuleList(
            [
                nn.Sequential(featurizers[member], classifiers[member])
                for member in range(self.num_members)
            ]
        )

        if self.hparams["shared_init"]:
            network_0 = self.networks[0]
            for i in range(1, self.num_members):
                network_i = self.networks[i]
                for param_0, param_i in zip(network_0.parameters(), network_i.parameters()):
                    param_i.data = param_0.data

        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.num_classes = num_classes

        # domain matcher
        self._init_optimizer()
        self._init_swa()
        self.soup = misc.Soup(self.networks)
        self._init_temperature()


    def _init_optimizer(self):
        lrs = [self.hparams["lr"] for _ in range(self.num_members)]
        if self.hparams.get("lr_ratio", 0) != 0:
            for member in range(self.num_members):
                lrs[member] /= float(self.hparams.get("lr_ratio"))**member
        print("Ensembling lrs: ", lrs)
        self.optimizers = [
            torch.optim.Adam(
                list(self.networks[member].parameters()),
                lr=lrs[member],
                weight_decay=self.hparams["weight_decay"],
            ) for member in range(self.num_members)
        ]

    def _init_swa(self):
        if self.hparams['swa']:
            assert self.hparams['swa'] == 1
            self.swas = [
                misc.SWA(self.networks[member], hparams=self.hparams)
                for member in range(self.num_members)
            ]
            self.soupswa = misc.Soup(
                networks=[swa.network_swa for swa in self.swas])
        else:
            self.swas = []

    def update(self, minibatches, unlabeled=None):
        if self.hparams['specialized'] == 1:
            nlls_per_member = self._update_specialized(minibatches)
        elif self.hparams['specialized']:
            # todo create diversity randomly per batch
            nlls_per_member = self._update_partial(minibatches, step=self.hparams['specialized'])
        else:
            nlls_per_member = self._update_full(minibatches)

        out = {"nll": torch.stack(nlls_per_member, dim=0).mean()}
        for key in range(self.num_members):
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

        for member in range(self.num_members):
            logits_member = self.networks[member](all_x)
            nll_member = F.cross_entropy(logits_member, all_classes, reduction="mean")
            nlls_per_member.append(nll_member)

            # optimization step
            self.optimizers[member].zero_grad()
            nll_member.backward()
            self.optimizers[member].step()

        return nlls_per_member

    def _update_partial(self, minibatches, step=2):
        num_domains_per_member = self.num_members // len(minibatches)
        index_per_member = [
            [num_domains_per_member * i + j
             for j in range(num_domains_per_member)]
            for i in range(self.num_members)
        ]

        nlls_per_member = []  # (num_members, num_domains_per_member)

        for member in range(self.num_members):
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
        assert self.num_members % len(minibatches) == 0
        num_domains_per_member = self.num_members // len(minibatches)
        index_per_member = [
            [num_domains_per_member * i + j
             for j in range(num_domains_per_member)]
            for i in range(self.num_members)
        ]
        x_per_member = [
            torch.cat([minibatches[index][0]
                       for index in index_per_member[i]])
            for i in range(self.num_members)
        ]
        classes_per_member = [
            torch.cat([minibatches[index][1]
                       for index in index_per_member[i]])
            for i in range(self.num_members)
        ]

        nlls_per_member = []  # (num_members, num_domains_per_member)

        for member in range(self.num_members):
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
        if self.hparams['swa']:
            self.soupswa.network_soup.to(device)
            for swa in self.swas:
                swa.network_swa.to(device)

    def eval(self):
        Algorithm.eval(self)
        self.soup.network_soup.eval()
        if self.hparams['swa']:
            self.soupswa.network_soup.eval()
            for swa in self.swas:
                swa.network_swa.eval()

    def train(self, *args):
        Algorithm.train(self, *args)
        self.soup.network_soup.train(*args)
        if self.hparams['swa']:
            self.soupswa.network_soup.train(*args)
            for swa in self.swas:
                swa.network_swa.train(*args)

    def predict(self, x):
        results = {}
        batch_logits = []
        batch_logits_swa = []

        for num_member in range(self.num_members):
            logits = self.networks[num_member](x)
            batch_logits.append(logits)
            results["net" + str(num_member)] = logits
            if self.hparams['swa']:
                logits_swa = self.swas[num_member].network_swa(x)
                batch_logits_swa.append(logits_swa)
                results["swa" + str(num_member)] = logits_swa
            results["soup"] = self.soup.network_soup(x)
            results["soupswa"] = self.soupswa.network_soup(x)

        results["net"] = torch.mean(torch.stack(batch_logits, dim=0), 0)
        if self.hparams['swa']:
            results["swa"] = torch.mean(torch.stack(batch_logits_swa, dim=0), 0)

        return results

    def accuracy(self, *args, **kwargs):
        return ERM.accuracy(self, *args, **kwargs)

    def _init_temperature(self):
        return ERM._init_temperature(self)

    def get_temperature(self, *args, **kwargs):
        return ERM.get_temperature(self, *args, **kwargs)
