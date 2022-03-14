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
    # "Fish",
    "IRM",
    "Subspace",
    "SWA",
    # "IRMAdv",
    # "GroupDRO",
    # "Mixup",
    # "MLDG",
    # "CORAL",
    # "COREL",
    # "MMD",
    # "DANN",
    # "CDANN",
    # "MTL",
    # "SagNet",
    # "ARM",
    # "VREx",
    # "VRExema",
    # "RSC",
    # "SD",
    # "ANDMask",
    # "SANDMask",
    # "IGA",
    # "SelfReg",
    # "FisherMMD",
    # "LFF",
    # "KernelDiversity",
    # "EnsembleKernelDiversity",
    # "Fishr",
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
            if self.hparams['swa'] != 1:
                self.swa_temperatures = []
                self.t_swa_optimizers = []
                for _ in range(self.hparams['swa']):
                    self.swa_temperatures.append(nn.Parameter(torch.ones(1), requires_grad=True))
                    self.t_swa_optimizers = torch.optim.Adam(
                        [self.swa_temperatures[-1]],
                        lr=self.hparams["lr"],
                        weight_decay=self.hparams["weight_decay"],
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
        if self.hparams['swa'] == 1:
            if return_optim:
                return None, None
            return None
        if not key.startswith("swa"):
            raise ValueError()
        i = int(key[-1])
        if return_optim:
            return self.swa_temperatures[i], self.t_swa_optimizers[i]
        return self.swa_temperatures[i]

    def _init_swa(self):
        if self.hparams['swa']:
            if self.hparams['swa'] == 1:
                self.swa = misc.SWA(self.network, hparams=self.hparams)
            else:
                self.swas = [misc.SWA(self.network, hparams=self.hparams) for _ in range(self.hparams['swa'])]
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
            if self.hparams['swa'] == 1:
                self.swa.update()
            else:
                for swa in self.swas:
                    swa.update()

        return {key: value.item() for key, value in output_dict.items()}

    def predict(self, x):
        results = {"net": self.network(x)}
        if self.hparams['swa']:
            if self.hparams['swa'] == 1:
                results["swa"] = self.swa.network_swa(x)
            else:
                batch_logits_swa = []
                for i in range(self.hparams['swa']):
                    logits_swa = self.swas[i].network_swa(x)
                    results["swa" + str(i)] = logits_swa
                    batch_logits_swa.append(logits_swa)

                results["swa"] = torch.mean(torch.stack(batch_logits_swa, dim=0), 0)
        return results

    # def predict_feat(self, x):
    #     feats_network = self.featurizer(x)
    #     if self.hparams['swa']:
    #         feats_swa = self.swa.get_featurizer()(x)
    #         results = {"swa": feats_swa, "net": feats_network}
    #     else:
    #         results = {"net": feats_network}
    #     return results

    def eval(self):
        Algorithm.eval(self)
        if self.hparams['swa']:
            if self.hparams['swa'] == 1:
                self.swa.network_swa.eval()
            else:
                for i in range(self.hparams['swa']):
                    self.swas[i].network_swa.eval()

    def train(self, *args):
        Algorithm.train(self, *args)
        if self.hparams['swa']:
            if self.hparams['swa'] == 1:
                self.swa.network_swa.train(*args)
            else:
                for i in range(self.hparams['swa']):
                    self.swas[i].network_swa.train(*args)

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

                    if key in ["net", "swa", "swa0", "swa1"]:
                        temperature = self.get_temperature(key)
                        if temperature is None:
                            continue
                        temperature = temperature.to(device)
                        probstemp = torch.softmax(
                            misc.apply_temperature_on_logits(logits.detach(), temperature),
                            dim=1
                        )
                        if "confstemp" not in dict_stats[key]:
                            dict_stats[key]["confstemp"] = []
                        dict_stats[key]["confstemp"].append(probstemp.max(dim=1)[0].cpu())

        for key0 in dict_stats:
            for key1 in dict_stats[key0]:
                dict_stats[key0][key1] = torch.cat(dict_stats[key0][key1])

        results = {}
        for key in dict_stats:
            results[f"Accuracies/acc_{key}"] = sum(dict_stats[key]["correct"].numpy()) / \
                                               len(dict_stats[key]["correct"].numpy())
            results[f"Calibration/ece_{key}"] = misc.get_ece(
                dict_stats[key]["confs"].numpy(), dict_stats[key]["correct"].numpy()
            )
            if "confstemp" in dict_stats[key]:
                results[f"Calibration/ecetemp_{key}"] = misc.get_ece(
                    dict_stats[key]["confstemp"].numpy(), dict_stats[key]["correct"].numpy()
                )

        targets_torch = torch.cat(batch_classes)
        for regex in ["swanet", "swa0net0", "swa0net", "swa01", "net01"]:
            if regex == "swanet":
                key0 = "swa"
                key1 = "net"
            elif regex == "swa0net0":
                key0 = "swa0"
                key1 = "net0"
            elif regex == "swa0net":
                key0 = "swa0"
                key1 = "net"
            elif regex == "swa01":
                key0 = "swa0"
                key1 = "swa1"
            elif regex == "net01":
                key0 = "net0"
                key1 = "net1"
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
            results[f"Diversity/{regex}agre"] = diversity_metrics.agreement_measure(
                targets, preds0, preds1
            )
            # results[f"Diversity/{regex}doublefault"] = diversity_metrics.double_fault(targets, preds0, preds1)
            # results[f"Diversity/{regex}singlefault"] = diversity_metrics.single_fault(targets, preds0, preds1)
            results[f"Diversity/{regex}qstat"] = diversity_metrics.Q_statistic(
                targets, preds0, preds1
            )

            # new div metrics
            probs0 = dict_stats[key0]["probs"].numpy()
            probs1 = dict_stats[key1]["probs"].numpy()
            results[f"Diversity/{regex}l2"] = diversity_metrics.l2(probs0, probs1)
            results[f"Diversity/{regex}nd"] = diversity_metrics.normalized_disagreement(
                targets, probs0, probs1
            )

        # Flatness metrics
        if compute_trace and hessian is not None:
            # feats0 = dict_stats[key0]["feats"]
            # hessian_comp_swa = hessian(
            #     self.swa.get_classifier(), nn.CrossEntropyLoss(reduction='sum'), data=(feats0, targets_torch), cuda=True)
            if self.hparams['swa'] == 1:
                hessian_comp_swa = hessian(
                    self.swa.network_swa,
                    nn.CrossEntropyLoss(reduction='sum'),
                    dataloader=loader,
                    cuda=True
                )
                results[f"Flatness/swatrace"] = np.mean(hessian_comp_swa.trace())
            else:
                hessian_comp_swa = hessian(
                    self.swas[0].network_swa,
                    nn.CrossEntropyLoss(reduction='sum'),
                    dataloader=loader,
                    cuda=True
                )
                results[f"Flatness/swa0trace"] = np.mean(hessian_comp_swa.trace())

            hessian_comp_net = hessian(
                self.network,
                nn.CrossEntropyLoss(reduction='sum'),
                dataloader=loader,
                cuda=True
            )
            results[f"Flatness/nettrace"] = np.mean(hessian_comp_net.trace())

        assert self.swa_temperature.requires_grad
        if update_temperature:
            for _ in range(50):
                for key in ["net", "swa", "swa0", "swa1"]:
                    if key not in dict_stats:
                        continue
                    logits = dict_stats[key]["logits"].to(device)
                    temperature, optimizer = self.get_temperature(
                        key, return_optim=True)
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
                    output_dict.update({key+str(i): value for key, value in swa_dict.items()})

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
        self.featurizers = nn.ModuleList(
            [networks.Featurizer(input_shape, self.hparams) for _ in range(self.num_members)]
        )
        self.features_size = self.featurizers[0].n_outputs
        self.classifiers = nn.ModuleList(
            [
                extend(
                    networks.Classifier(
                        self.features_size,
                        num_classes,
                        self.hparams["nonlinear_classifier"],
                        hparams=self.hparams,
                    )
                ) for _ in range(self.num_members)
            ]
        )
        print(self.classifiers)

        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.num_classes = num_classes

        # member diversifier
        if self.hparams["diversity_loss"] in diversity.DICT_NAME_TO_DIVERSIFIER:
            self.member_diversifier = diversity.DICT_NAME_TO_DIVERSIFIER[
                self.hparams["diversity_loss"]
            ](hparams=self.hparams, features_size=self.features_size, num_classes=self.num_classes)
        else:
            self.member_diversifier = None

        # domain matcher
        self._init_optimizer()
        self.register_buffer("update_count", torch.tensor([0]))
        self.init_domain_matcher()
        self._init_swa()

    def init_domain_matcher(self):
        if self.hparams["similarity_loss"] == "none":
            self.domain_matchers = None
        elif self.hparams["similarity_loss"] == "fishr":
            raise ValueError()
            # self.domain_matchers = [
            #     FishrDomainMatcher(self.hparams, self.optimizer, self.num_domains)
            #     for _ in range(self.num_members)
            # ]
        else:
            raise ValueError(self.hparams["similarity_loss"])

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.featurizers.parameters()) + list(self.classifiers.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def _init_swa(self):
        if self.hparams['swa']:
            assert self.hparams['swa'] == 1
            self.swas = [
                misc.SWA(
                    nn.Sequential(self.featurizers[num_member], self.classifiers[num_member]),
                    hparams=self.hparams
                ) for num_member in range(self.num_members)
            ]
        else:
            self.swas = []

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_classes = torch.cat([y for x, y in minibatches])
        bsize = minibatches[0][0].size(0)

        features_per_member = []  # (num_classifiers, num_minibatches, bsize, n_outputs_featurizer)
        logits_per_member = []  # (num_classifiers, num_minibatches, bsize, num_classes)
        nlls_per_member = []  # (num_classifiers, num_minibatches)

        for member in range(self.num_members):
            features_member = self.featurizers[member](all_x)
            logits_member = self.classifiers[member](features_member)

            features_per_member.append(features_member)
            logits_per_member.append(logits_member)
            nll_member = F.cross_entropy(logits_member, all_classes, reduction="none")
            nlls_per_member.append(nll_member)

        all_nll = torch.stack(nlls_per_member, dim=0).sum(0).mean()
        objective = 0 + all_nll
        out = {"nll": (all_nll).item()}
        for key in range(self.num_members):
            out[f"nll_{key}"] = nlls_per_member[key].mean().item()

        penalty_active = self.update_count >= self.hparams["penalty_anneal_iters"]
        if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
            # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
            # gradient magnitudes that happens at this step.
            self._init_optimizer()
        self.update_count += 1

        # domain matching
        if self.domain_matchers is not None:
            if self.hparams["similarity_loss"] == "fishr":
                for member in range(self.num_members):
                    logits_member = logits_per_member[member]
                    dict_penalty = self.domain_matchers[member].compute_fishr_penalty(
                        logits_member, all_classes=all_classes, classifier=self.classifiers[member]
                    )
                    if penalty_active:
                        objective = objective + (
                            self.hparams["lambda_domain_matcher"] * dict_penalty["penalty_var"]
                        )
                    out.update(
                        {f'{key}_{member}': value.item() for key, value in dict_penalty.items()}
                    )
            else:
                raise ValueError(self.domain_matcher)

        # firstorder information bottleneck
        ibstats_per_member = (
            logits_per_member if self.hparams["ib_space"] == "logits" else features_per_member
        )
        loss_ib_firstorder = 0
        for member in range(self.num_members):
            ibstats_per_member_domain = ibstats_per_member[member].reshape(
                self.num_domains, bsize, -1
            )
            for domain in range(self.num_domains):
                loss_ib_firstorder += ibstats_per_member_domain[domain].var(dim=0).mean()
        if penalty_active:
            objective = objective + (
                loss_ib_firstorder * self.hparams["lambda_ib_firstorder"] /
                (self.num_members * self.num_domains)
            )

        # diversity across members
        if self.member_diversifier is not None and self.num_members > 1:
            kwargs = {
                "features_per_member":
                torch.stack(features_per_member, dim=0
                           ).reshape(self.num_members, self.num_domains, bsize, self.features_size),
                "logits_per_member":
                torch.stack(logits_per_member, dim=0
                           ).reshape(self.num_members, self.num_domains, bsize, self.num_classes),
                "classes":
                all_classes,
                "nlls_per_member":
                torch.stack(nlls_per_member,
                            dim=0).reshape(self.num_members, self.num_domains, bsize),
                "classifiers":
                self.classifiers
            }
            dict_diversity = self.member_diversifier.forward(**kwargs)
            out.update({key: value.item() for key, value in dict_diversity.items()})
            if penalty_active:
                objective = objective + dict_diversity["loss_div"
                                                      ] * self.hparams["lambda_diversity_loss"]

        # optim steps
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        if self.hparams['swa']:
            for swa in self.swas:
                swa.update()
        return out

    def eval(self):
        Algorithm.eval(self)
        if self.hparams['swa']:
            for swa in self.swas:
                swa.network_swa.eval()

    def train(self, *args):
        Algorithm.train(self, *args)
        if self.hparams['swa']:
            for swa in self.swas:
                swa.network_swa.train(*args)

    def predict(self, x):
        results = {}
        batch_logits = []
        batch_logits_swa = []

        for num_member in range(self.num_members):
            features = self.featurizers[num_member](x)
            logits = self.classifiers[num_member](features)
            batch_logits.append(logits)
            results["net" + str(num_member)] = logits
            if self.hparams['swa']:
                logits_swa = self.swas[num_member].network_swa(x)
                batch_logits_swa.append(logits_swa)
                results["swa" + str(num_member)] = logits_swa

        results["net"] = torch.mean(torch.stack(batch_logits, dim=0), 0)
        if self.hparams['swa']:
            results["swa"] = torch.mean(torch.stack(batch_logits_swa, dim=0), 0)

        return results

    def accuracy(self, *args, **kwargs):
        return ERM.accuracy(self, *args, **kwargs)
