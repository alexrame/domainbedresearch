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

    def predict(self, x):
        raise NotImplementedError


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

    # def _init_swa(self):
    #     if self.hparams['swa']:
    #         self.swa = misc.SWA(self.network, hparams=self.hparams)
    #     else:
    #         self.swa = None

    # def predict(self, x):
    #     results = {"net": self.network(x)}
    #     if self.hparams['swa']:
    #         results["swa"] = self.swa.network_swa(x)
    #     return results

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
