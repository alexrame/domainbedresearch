import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from domainbed import losses

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


class KLPreds(DiversityLoss):
    def forward(self, logits_per_member, **kwargs):
        num_members = logits_per_member.size(0)
        num_preds = logits_per_member.size(-1)
        probs = torch.softmax(
            logits_per_member.reshape(num_members, -1, num_preds).transpose(0, 1), dim=2)
        # Probs = predicted probabilites on target batch.
        B, H, D = probs.shape # B=batch_size, H=heads, D=pred_dim
        marginal_p = probs.mean(dim=0)
        # H, D
        marginal_p = torch.einsum("hd,ge->hgde", marginal_p, marginal_p)
        # H, H, D, D marginal_p = rearrange(marginal_p, "h g d e -> (h g) (d e)") # H^2, D^2
        joint_p = torch.einsum("bhd,bge->bhgde", probs, probs).mean(dim=0)
        # H, H, D, D joint_p = rearrange(joint_p, "h g d e -> (h g) (d e)") # H^2, D^2
        kl_divs = joint_p * (joint_p.log() - marginal_p.log())
        kl_grid = rearrange(kl_divs.sum(dim=-1), "(h g) -> h g", h=H)
        # H, H
        pairwise_mis = torch.triu(kl_grid, diagonal=1)
        # Get only off-diagonal KL divergences
        return pairwise_mis.mean()




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
            nll_01 = losses.SoftCrossEntropyLoss()(
                input=logits_per_member[0][domain],
                target=torch.round(torch.softmax(logits_per_member[1][domain], dim=1))
            )
            batch_nll_01.append(nll_01)
            nll_10 = losses.SoftCrossEntropyLoss()(
                input=logits_per_member[1][domain],
                target=torch.round(torch.softmax(logits_per_member[0][domain], dim=1))
            )
            loss = (nll_01 - nlls_per_member[0][domain].mean()
                   )**2 + (nll_10 - nlls_per_member[1][domain].mean())**2
        mean_nll_01 = torch.stack(batch_nll_01, dim=0).mean(dim=0)
        return {"loss_div": loss, "nll_01": mean_nll_01}
