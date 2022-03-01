import torch
from domainbed.lib.diversity.standard_diversity import DiversityLoss


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
