import torch

import torch.nn as nn
import torch.nn.functional as F


class DiversityLoss(torch.nn.Module):
    diversity_type = "regularization"

    def __init__(self, hparams, **kwargs):
        self.hparams = hparams
        super(DiversityLoss, self).__init__()

    def forward(self, **kwargs):
        """
        Args:
            logits : tensor of shape (num_members, num_envs, bsize, num_classes)
            losses: tensor of shape (num_members, num_envs, bsize) containing losses for each element
            classifiers: ModuleList containing all classifiers.
        """
        raise NotImplementedError


class GroupDRO(DiversityLoss):
    diversity_type = "sampling"
    def __init__(self, hparams, num_domains, **kwargs):
        DiversityLoss.__init__(self, hparams)
        self.num_domains = num_domains
        self.q = torch.ones(num_domains)

    def compute_weighted_loss(self, active_loss, sampling_loss):
        sampling_losses = sampling_loss.reshape(self.num_domains, -1).mean(dim=1)
        for loss in sampling_losses:
            self.q *= (self.hparams["div_eta"] * loss.detach()).exp()

        self.q /= self.q.sum()

        active_losses = active_loss.reshape(self.num_domains, -1).mean(dim=1)
        loss_weighted = torch.dot(active_losses, self.q)

        return loss_weighted


class Bagging(DiversityLoss):
    diversity_type = "sampling"
    def __init__(self, hparams, num_domains, **kwargs):
        DiversityLoss.__init__(self, hparams)
        self.num_domains = num_domains

    def compute_weighted_loss(self, active_loss, sampling_loss):
        q = (self.hparams["div_eta"] * sampling_loss.detach()).exp()
        q /= q.sum()
        loss_weighted = torch.dot(active_loss, q)

        return loss_weighted


class BaggingPerDomain(DiversityLoss):
    diversity_type = "sampling"

    def __init__(self, hparams, num_domains, **kwargs):
        DiversityLoss.__init__(self, hparams)
        self.num_domains = num_domains

    def compute_weighted_loss(self, active_loss, sampling_loss):
        q = (self.hparams["div_eta"] * sampling_loss.detach()).exp().reshape(self.num_domains, -1)
        q_sum_per_domain = q.sum(dim=1, keepdim=True)
        final_q = (q / (self.num_domains * q_sum_per_domain)).view((-1, ))
        loss_weighted = torch.dot(active_loss, final_q)

        return loss_weighted


class LogitDistance(DiversityLoss):

    def forward(self, logits_per_member, **kwargs):
        """Diversity in logits distance
        Args:
            logits_per_member : tensors of shape (num_classifiers, num_minibatches, bsize, num_classes)
        """
        loss = 0
        num_members = logits_per_member.size(0)
        num_domains = logits_per_member.size(1)
        for domain in range(num_domains):
            logits = logits_per_member[:, domain].reshape((num_members, -1))
            distances = F.pdist(logits, p=2)
            loss -= distances.mean()
        return {"loss_div": loss}



class SoftCrossEntropyLoss(nn.modules.loss._Loss):

    def forward(self, input, target):
        """
        Cross entropy that accepts soft targets
        Args:
            pred: predictions for neural network
            targets: targets, can be soft
            size_average: if false, sum is returned instead of mean
        Examples::
            input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
            input = torch.autograd.Variable(out, requires_grad=True)
            target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
            target = torch.autograd.Variable(y1)
            loss = cross_entropy(input, target)
            loss.backward()
        """
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))


class CEDistance(DiversityLoss):

    def forward(self, logits_per_member, nlls_per_member, **kwargs):
        """Diversity in logits distance
        Args:
            logits_per_member : list of shape (num_classifiers, num_minibatches, bsize, num_classes)
        """
        loss = 0
        num_members = logits_per_member.size(0)
        # assert num_members == 2
        num_domains = logits_per_member.size(1)
        batch_nll_01 = []
        for domain in range(num_domains):
            nll_01 = SoftCrossEntropyLoss()(
                input=logits_per_member[0][domain],
                target=torch.round(torch.softmax(logits_per_member[1][domain], dim=1))
            )
            batch_nll_01.append(nll_01)
            nll_10 = SoftCrossEntropyLoss()(
                input=logits_per_member[1][domain],
                target=torch.round(torch.softmax(logits_per_member[0][domain], dim=1))
            )
            loss = (nll_01 - nlls_per_member[0][domain].mean()
                   )**2 + (nll_10 - nlls_per_member[1][domain].mean())**2
        mean_nll_01 = torch.stack(batch_nll_01, dim=0).mean(dim=0)
        return {"loss_div": loss, "nll_01": mean_nll_01}
