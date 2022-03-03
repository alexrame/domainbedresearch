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
from domainbed.lib import misc, diversity_metrics, diversity, sam, sammav, losses
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
    "Fish",
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
    "Fishr",
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
        self.classifier = networks.Classifier(self.featurizer.n_outputs, num_classes, self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self._init_mav()
        self._init_optimizer()

    def _init_mav(self):
        if self.hparams['mav']:
            self.mav = misc.MovingAvg(self.network, hparams=self.hparams)
        else:
            self.mav = None

    def _init_optimizer(self):
        if self.hparams.get('sam'):
            phosam = 10 * self.hparams["phosam"] if self.hparams["samadapt"] else self.hparams[
                "phosam"]
            if self.hparams.get('sam') == "inv":
                phosam = - phosam
            if self.hparams.get('sam') == "onlymav":
                self.optimizer = sammav.SAMMAV(
                    self.network.parameters(),
                    params_mav=self.mav.network_mav.parameters(),
                    base_optimizer=torch.optim.Adam,
                    adaptive=self.hparams["samadapt"],
                    rho=phosam,
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams["weight_decay"],
                )
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
        elif self.hparams['sam'] == "mav":
            predictions = self.network(all_x)
            with torch.no_grad():
                predictions_mav = self.mav.network_mav(all_x)
            loss = F.cross_entropy((predictions + predictions_mav) / 2, all_classes)
        elif self.hparams['sam'] == "mavsam":
            predictions = self.network(all_x)
            with torch.no_grad():
                predictions_mav = self.mav.network_mav(all_x)
            loss = (1 + self.hparams['mavsamcoeff']) * F.cross_entropy(
                (predictions + self.hparams['mavsamcoeff'] * predictions_mav) /
                (1 + self.hparams['mavsamcoeff']), all_classes
            )
        elif self.hparams['sam'] == "onlymav":
            loss = F.cross_entropy(self.mav.network_mav(all_x), all_classes)
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

        if self.hparams['mav']:
            self.mav.update()
        return {key: value.item() for key, value in output_dict.items()}

    def predict(self, x):
        preds_network = self.network(x)
        if self.hparams['mav']:
            preds_mav = self.mav.network_mav(x)
            results = {"mav": preds_mav, "net": preds_network}
        else:
            results = {"net": preds_network}
        return results

    def predict_feat(self, x):
        feats_network = self.featurizer(x)
        if self.hparams['mav']:
            feats_mav = self.mav.get_featurizer()(x)
            results = {"mav": feats_mav, "net": feats_network}
        else:
            results = {"net": feats_network}
        return results

    def eval(self):
        Algorithm.eval(self)
        if self.hparams['mav']:
            self.mav.network_mav.eval()

    def train(self, *args):
        Algorithm.train(self, *args)
        if self.hparams['mav']:
            self.mav.network_mav.train(*args)

    def accuracy(self, loader, device, compute_trace=False):
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
                        dict_stats[key] = {"preds": [], "confs": [], "correct": []}  # , "feats": []}
                        if os.environ.get("DIVMETRICS"):
                            dict_stats[key]["probs"] = []
                    logits = dict_logits[key]
                    try:
                        preds = logits.argmax(1)
                    except:
                        pdb.set_trace()
                    if os.environ.get("DIVMETRICS"):
                        dict_stats[key]["probs"].append(torch.softmax(logits, dim=1))

                    dict_stats[key]["preds"].append(preds.cpu())
                    # dict_stats[key]["feats"].append(dict_feats[key])
                    dict_stats[key]["confs"].append(logits.max(1)[0].cpu())
                    dict_stats[key]["correct"].append(preds.eq(y).float().cpu())

        for key0 in dict_stats:
            for key1 in dict_stats[key0]:
                dict_stats[key0][key1] = torch.cat(dict_stats[key0][key1])

        results = {}
        for key in dict_stats:
            results[f"Accuracies/acc_{key}"] = sum(dict_stats[key]["correct"].numpy()) / \
                                               len(dict_stats[key]["correct"].numpy())
            results[f"Calibration/ece_{key}"] = misc.get_ece(
                dict_stats[key]["confs"].numpy(), dict_stats[key]["correct"].numpy())

        for regex in ["mavnet", "mavnet0", "mav01", "net01"]:
            if regex == "mavnet":
                key0 = "mav"
                key1 = "net"
            elif regex == "mavnet0":
                key0 = "mav0"
                key1 = "net0"
            elif regex == "mav01":
                key0 = "mav0"
                key1 = "mav1"
            elif regex == "net01":
                key0 = "net0"
                key1 = "net1"
            else:
                raise ValueError(regex)

            if key0 not in dict_stats:
                continue
            assert key1 in dict_stats
            targets_torch = torch.cat(batch_classes)
            targets = targets_torch.cpu().numpy()
            preds0 = dict_stats[key0]["preds"].cpu().numpy()
            preds1 = dict_stats[key1]["preds"].cpu().numpy()
            results[f"Diversity/{regex}ratio"] = diversity_metrics.ratio_errors(targets, preds0, preds1)
            results[f"Diversity/{regex}agre"] = diversity_metrics.agreement_measure(targets, preds0, preds1)
            # results[f"Diversity/{regex}doublefault"] = diversity_metrics.double_fault(targets, preds0, preds1)
            # results[f"Diversity/{regex}singlefault"] = diversity_metrics.single_fault(targets, preds0, preds1)
            results[f"Diversity/{regex}qstat"] = diversity_metrics.Q_statistic(targets, preds0, preds1)

            if os.environ.get("DIVMETRICS"):
                probs0 = dict_stats[key0]["probs"].cpu().numpy()
                probs1 = dict_stats[key1]["probs"].cpu().numpy()
                results[f"Diversity/{regex}l2"] = diversity_metrics.l2(
                    probs0, probs1)
                results[f"Diversity/{regex}nd"] = diversity_metrics.normalized_disagreement(
                    targets, probs0, probs1)

            # Flatness metrics
            if compute_trace and regex == "mavnet" and hessian is not None:
                # feats0 = dict_stats[key0]["feats"]
                # hessian_comp_mav = hessian(
                #     self.mav.get_classifier(), nn.CrossEntropyLoss(reduction='sum'), data=(feats0, targets_torch), cuda=True)
                hessian_comp_mav = hessian(
                    self.mav.network_mav, nn.CrossEntropyLoss(reduction='sum'), dataloader=loader, cuda=True)
                results[f"Flatness/{key0}trace"] = np.mean(hessian_comp_mav.trace())

                # feats1 = dict_stats[key1]["feats"]
                # hessian_comp_net = hessian(
                #     self.classifier, nn.CrossEntropyLoss(reduction='sum'), data=(feats1, targets_torch), cuda=True)
                hessian_comp_net = hessian(
                    self.network, nn.CrossEntropyLoss(reduction='sum'), dataloader=loader, cuda=True)
                results[f"Flatness/{key1}trace"] = np.mean(hessian_comp_net.trace())
        self.train()
        return results


class Subspace(ERM):
    """
    Subspace learning
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Subspace, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(self.featurizer.n_outputs, num_classes, self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.net = {}
        self.size_code = 5
        self.hypernet = nn.Linear(self.size_code, count_param(self.network))
        self.optimizer = torch.optim.Adam(
            self.hypernet.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])
        set_requires_grad(self.network, False)
        self._init_mav()

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_classes = torch.cat([y for x, y in minibatches])

        net_copy = copy.deepcopy(self.network)
        set_requires_grad(net_copy, False)

        code = torch.rand(self.size_code).to("cuda")
        param_hyper = self.hypernet(code)
        count_p = 0
        for pnet in net_copy.parameters():
            phyper = param_hyper[count_p: count_p + int(pnet.numel())].reshape(*pnet.shape)
            pnet.copy_(phyper)
            count_p += int(pnet.numel())
        loss_reg = (torch.norm(self.hypernet.weight, dim=1)).sum()
        loss = F.cross_entropy(net_copy(all_x), all_classes) + self.hparams["penalty_reg"] * loss_reg

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        param_hyper = self.hypernet.bias
        count_p = 0
        for pnet in self.network.parameters():
            phyper = param_hyper[count_p: count_p + int(pnet.numel())].reshape(*pnet.shape)
            pnet.copy_(phyper)
            count_p += int(pnet.numel())
        return {"net": self.network(x)}


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
            if self.hparams["div_data"] == "uda":
                assert unlabeled is not None
                div_objective, div_dict = self.increase_diversity_unlabeled(unlabeled)
            else:
                div_objective, div_dict = self.increase_diversity_labeled(all_x, all_features, all_classes, all_loss, all_logits, bsize)
            if penalty_active:
                objective = objective + div_objective
            output_dict.update(div_dict)
        else:
            pass

        if self.hparams.get('sam'):
            self.optimizer.zero_grad()
            # first forward-backward pass
            objective.backward()
            self.optimizer.first_step(zero_grad=True)
            # second forward-backward pass
            objective2 = F.cross_entropy(
                self.classifier(self.featurizer(all_x)),
                all_classes, reduction="mean")
            # make sure to do a full forward pass
            objective2.backward()
            self.optimizer.second_step()
            output_dict["objective2"] = objective2
        else:
            self.optimizer.zero_grad()
            objective.backward()
            self.optimizer.step()
        if self.hparams['mav']:
            self.mav.update()
        return {key: value.item() for key, value in output_dict.items()}

    def increase_diversity_unlabeled(self, unlabeled):
        bsize = unlabeled[0].size(0)
        num_domains = len(unlabeled)
        all_x = torch.cat(unlabeled)
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        with torch.no_grad():
            mav_features = self.mav.get_featurizer()(all_x)
            mav_logits = self.mav.get_classifier()(all_features)
        assert self.member_diversifier.diversity_type != "sampling"
        kwargs = {
            "features_per_member":
            torch.stack([all_features, mav_features], dim=0).reshape(2, num_domains, bsize, self.features_size),
            "logits_per_member":
            torch.stack([all_logits, mav_logits], dim=0).reshape(2, num_domains, bsize, self.num_classes),
            "classes": None,
            "nlls_per_member": None
        }
        dict_diversity = self.member_diversifier.forward(**kwargs)

        objective = dict_diversity["loss_div"] * self.hparams["lambda_diversity_loss"]
        if self.hparams["lambda_entropy"]:
            objective = objective + self.hparams["lambda_entropy"] * losses.entropy_regularizer(all_logits)
        return objective, dict_diversity

    def increase_diversity_labeled(self,all_x, all_features, all_classes, all_loss, all_logits, bsize):
        with torch.no_grad():
            mav_features = self.mav.get_featurizer()(all_x)
            mav_logits = self.mav.get_classifier()(all_features)
            mav_loss = F.cross_entropy(mav_logits, all_classes, reduction="none")
        output_dict = {}
        if self.member_diversifier.diversity_type == "sampling":
            loss_weighted = self.member_diversifier.compute_weighted_loss(
                active_loss=all_loss, sampling_loss=mav_loss
            )
            output_dict["lossw"] = loss_weighted
            objective = loss_weighted - all_loss.mean()
        else:
            kwargs = {
                "features_per_member":
                torch.stack([all_features, mav_features], dim=0).reshape(2, self.num_domains, bsize, self.features_size),
                "logits_per_member":
                torch.stack([all_logits, mav_logits], dim=0).reshape(2, self.num_domains, bsize, self.num_classes),
                "classes":
                all_classes,
                "nlls_per_member":
                torch.stack([all_loss, mav_loss], dim=0).reshape(2, self.num_domains, bsize),
                # "classifiers": [self.classifier, self.mav.get_classifier()]
            }
            dict_diversity = self.member_diversifier.forward(**kwargs)
            output_dict.update(dict_diversity)
            objective = dict_diversity["loss_div"] * self.hparams["lambda_diversity_loss"]

        if self.hparams["lambda_entropy"]:
            objective = objective + self.hparams["lambda_entropy"] * losses.entropy_regularizer(all_logits)
        return objective, output_dict

class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.0).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (
            self.hparams["irm_lambda"]
            if self.update_count >= self.hparams["irm_penalty_anneal_iters"] else 1.0
        )
        nll = 0.0
        penalty = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams["irm_penalty_anneal_iters"]:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class Fish(ERM):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains, hparams)

        Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self._init_optimizer()
        self.optimizer_inner_state = None
        self._init_mav()

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(
            self.input_shape, self.num_classes, self.hparams, weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = misc.ParamDict(meta_weights)
        inner_weights = misc.ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)
        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"],
        )
        self.network.reset_weights(meta_weights)
        if self.hparams['mav']:
            self.mav.update()
        return {"loss": loss.item()}


class FishrDomainMatcher():
    def __init__(self, hparams, optimizer, num_domains):
        self.hparams = hparams
        self.optimizer = optimizer
        self.num_domains = num_domains

        self.loss_extended = extend(nn.CrossEntropyLoss(reduction='sum'))
        self.ema_per_domain_mean = [
            misc.MovingAverage(ema=hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self.ema_per_domain_var = [
            misc.MovingAverage(ema=hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]

        if self.hparams.get('mav') == "gradvar":
            raise NotImplementedError
            self.ema_per_domain_mean_mav = [
                misc.MovingAverage(ema=hparams["ema"], oneminusema_correction=True)
                for _ in range(self.num_domains)
            ]
            self.ema_per_domain_var_mav = [
                misc.MovingAverage(ema=hparams["ema"], oneminusema_correction=True)
                for _ in range(self.num_domains)
            ]

        self.list_methods = hparams["method"].split("_")

    def compute_fishr_penalty(self, all_logits, all_classes, classifier):
        grads = self._compute_grads(all_logits, all_classes, classifier)
        grads_mean_per_domain, grads_var_per_domain = self._compute_grads_mean_var(
            grads, self.ema_per_domain_mean, self.ema_per_domain_var
        )
        return {
            "penalty_mean": self._compute_distance(grads_mean_per_domain),
            "penalty_var": self._compute_distance(grads_var_per_domain)
        }

    def compute_fishr_penalty_mav(self, all_logits, all_classes, classifier, all_logits_mav, classifier_mav):
        grads = self._compute_grads(all_logits, all_classes, classifier)
        grads_mean_per_domain, grads_var_per_domain = self._compute_grads_mean_var(
            grads, self.ema_per_domain_mean, self.ema_per_domain_var
        )

        grads_mav = self._compute_grads(
            all_logits_mav, all_classes, classifier_mav, create_graph=False
        )
        grads_mean_per_domain_mav, grads_var_per_domain_mav = self._compute_grads_mean_var(
            grads_mav.detach(), self.ema_per_domain_mean_mav, self.ema_per_domain_var_mav
        )
        return {
            "penalty_mean":
            self._compute_distance_mav(grads_mean_per_domain, grads_mean_per_domain_mav),
            "penalty_var":
            self._compute_distance_mav(grads_var_per_domain, grads_var_per_domain_mav)
        }

    def _compute_grads(self, logits, y, classifier, create_graph=True):
        bsize = logits.size(0)
        self.optimizer.zero_grad()
        loss = self.loss_extended(logits, y)
        # calling first-order derivatives in the classifier while maintaining the per-sample gradients for all domains simultaneously
        with backpack(BatchGrad()):
            loss.backward(inputs=list(classifier.parameters()), retain_graph=True, create_graph=create_graph)
        dict_grads = {
            name: weights.grad_batch.clone().view(bsize, -1)
            for name, weights in classifier.named_parameters()
        }
        grads = torch.cat([dict_grads[key] for key in sorted(dict_grads.keys())], dim=1)
        return grads

    def _compute_grads_mean_var(self, grads, ema_per_domain_mean, ema_per_domain_var):
        # gradient variances per domain
        grads_mean_per_domain = []
        grads_var_per_domain = []

        bsize = grads.size(0) // self.num_domains
        for domain_id in range(self.num_domains):
            domain_grads = grads[domain_id * bsize:(domain_id + 1) * bsize]
            domain_mean = domain_grads.mean(dim=0, keepdim=True)
            grads_mean_per_domain.append(domain_mean)
            if "notcentered" in self.list_methods:
                domain_grads_centered = domain_grads
            else:
                domain_grads_centered = domain_grads - domain_mean
            grads_var_per_domain.append((domain_grads_centered).pow(2).mean(dim=0))

        # moving average
        for domain_id in range(self.num_domains):
            grads_mean_per_domain[domain_id] = ema_per_domain_mean[domain_id].update_value(
                grads_mean_per_domain[domain_id]
            )
            grads_var_per_domain[domain_id] = ema_per_domain_var[domain_id].update_value(
                grads_var_per_domain[domain_id]
            )

        return grads_mean_per_domain, grads_var_per_domain

    def _compute_distance(self, grads_per_domain):
        # compute gradient variances averaged across domains
        grads_mean = torch.stack(grads_per_domain, dim=0).mean(dim=0)
        if "zerohessian" in self.list_methods:
            grads_mean = torch.zeros_like(grads_mean)

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += (grads_per_domain[domain_id] - grads_mean).pow(2).mean()
        return penalty / self.num_domains

    def _compute_distance_mav(self, grads_per_domain, grads_per_domain_mav):
        # compute gradient variances averaged across domains
        grads_mean = torch.stack(grads_per_domain_mav, dim=0).mean(dim=0).detach()
        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += (grads_per_domain[domain_id] - grads_mean).pow(2).mean()
        return penalty / self.num_domains


class Fishr(ERM):
    "Invariant Gradient Variances for Out-of-distribution Generalization"

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

        self._init_optimizer()
        self.register_buffer("update_count", torch.tensor([0]))
        self.domain_matcher = FishrDomainMatcher(hparams, self.optimizer, num_domains)
        self._init_mav()

    def update(self, minibatches, unlabeled=False):
        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_classes = torch.cat([y for x, y in minibatches])

        output_dict = self.get_loss(all_x, all_classes)

        if self.hparams['sam']:
            raise NotImplementedError
            self.optimizer.zero_grad()
            # first forward-backward pass
            if self.hparams['sam'] == "nll":
                output_dict["nll"].backward()
            else:
                output_dict["objective"].backward()
            self.optimizer.first_step()
            self.optimizer.zero_grad()
            # second forward-backward pass
            output_dict_second = self.get_loss(all_x, all_classes)
            # make sure to do a full forward pass
            output_dict_second["objective"].backward()
            self.optimizer.second_step()
            output_dict.update(
                {key + "_secondstep": value for key, value in output_dict_second.items()}
            )
        else:
            self.optimizer.zero_grad()
            output_dict["objective"].backward()
            self.optimizer.step()
        if self.hparams['mav']:
            self.mav.update()

        return {key: value.item() for key, value in output_dict.items()}

    def get_loss(self, all_x, all_classes):
        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)

        if self.hparams['mav'] == "gradvar":
            raise NotImplementedError
            all_logits_mav = self.mav.network_mav(all_x)
            dict_penalty = self.domain_matcher.compute_fishr_penalty_mav(
                all_logits,
                all_classes,
                classifier=self.classifier,
                all_logits_mav=all_logits_mav,
                classifier_mav=self.mav.get_classifier()
            )
        else:
            dict_penalty = self.domain_matcher.compute_fishr_penalty(
                all_logits, all_classes, classifier=self.classifier
            )
        all_nll = F.cross_entropy(all_logits, all_classes)

        penalty_active = float(self.update_count >= self.hparams["penalty_anneal_iters"])
        if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
            # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
            # gradient magnitudes that happens at this step.
            if self.hparams["lambda"] + self.hparams["lambdamean"] != 0:
                self._init_optimizer()

        self.update_count += 1

        penalty = (
            self.hparams["lambda"] * dict_penalty["penalty_var"] +
            self.hparams["lambdamean"] * dict_penalty["penalty_mean"]
        )
        objective = all_nll + penalty_active * penalty
        output_dict = {'objective': objective, 'nll': all_nll, "penalty": penalty}
        output_dict.update({key: value for key, value in dict_penalty.items()})
        return output_dict


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
            [extend(
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
                self.hparams["diversity_loss"]](
                    hparams=self.hparams,
                    features_size=self.features_size,
                    num_classes=self.num_classes
                )
        else:
            self.member_diversifier = None

        # domain matcher
        self._init_optimizer()
        self.register_buffer("update_count", torch.tensor([0]))
        self.init_domain_matcher()
        self._init_mav()

    def init_domain_matcher(self):
        if self.hparams["similarity_loss"] == "none":
            self.domain_matchers = None
        elif self.hparams["similarity_loss"] == "fishr":
            self.domain_matchers = [
                FishrDomainMatcher(self.hparams, self.optimizer, self.num_domains) for _ in range(self.num_members)
            ]
        else:
            raise ValueError(self.hparams["similarity_loss"])

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.featurizers.parameters()) + list(self.classifiers.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def _init_mav(self):
        if self.hparams['mav']:
            self.mavs = [
                misc.MovingAvg(
                    nn.Sequential(self.featurizers[num_member], self.classifiers[num_member]),
                    hparams=self.hparams)
                for num_member in range(self.num_members)
            ]
        else:
            self.mavs = []

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
            ibstats_per_member_domain = ibstats_per_member[member].reshape(self.num_domains, bsize, -1)
            for domain in range(self.num_domains):
                loss_ib_firstorder += ibstats_per_member_domain[domain].var(dim=0).mean()
        if penalty_active:
            objective = objective + (
                loss_ib_firstorder * self.hparams["lambda_ib_firstorder"] / (self.num_members * self.num_domains)
            )

        # diversity across members
        if self.member_diversifier is not None and self.num_members > 1:
            kwargs = {
                "features_per_member":
                torch.stack(features_per_member, dim=0).reshape(
                    self.num_members, self.num_domains, bsize, self.features_size
                ),
                "logits_per_member":
                torch.stack(logits_per_member, dim=0).reshape(
                    self.num_members, self.num_domains, bsize, self.num_classes
                ),
                "classes": all_classes,
                "nlls_per_member":
                torch.stack(nlls_per_member, dim=0).reshape(self.num_members, self.num_domains, bsize),
                "classifiers": self.classifiers
            }
            dict_diversity = self.member_diversifier.forward(**kwargs)
            out.update({key: value.item() for key, value in dict_diversity.items()})
            if penalty_active:
                objective = objective + dict_diversity["loss_div"] * self.hparams["lambda_diversity_loss"]

        # optim steps
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        if self.hparams['mav']:
            for mav in self.mavs:
                mav.update()
        return out

    def eval(self):
        Algorithm.eval(self)
        if self.hparams['mav']:
            for mav in self.mavs:
                mav.network_mav.eval()

    def train(self, *args):
        Algorithm.train(self, *args)
        if self.hparams['mav']:
            for mav in self.mavs:
                mav.network_mav.train(*args)

    def predict(self, x):
        results = {}
        batch_logits = []
        batch_logits_mav = []

        for num_member in range(self.num_members):
            features = self.featurizers[num_member](x)
            logits = self.classifiers[num_member](features)
            batch_logits.append(logits)
            results["net" + str(num_member)] = logits
            if self.hparams['mav']:
                logits_mav = self.mavs[num_member].network_mav(x)
                batch_logits_mav.append(logits_mav)
                results["mav" + str(num_member)] = logits_mav

        results["net"] = torch.mean(torch.stack(batch_logits, dim=0), 0)
        if self.hparams['mav']:
            results["mav"] = torch.mean(torch.stack(batch_logits_mav, dim=0), 0)

        return results

    def accuracy(self, loader, device, compute_trace=False):
        return ERM.accuracy(self, loader, device, compute_trace)
