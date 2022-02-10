import torch.nn as nn
import torch.nn.functional as F
from backpack import backpack, extend
from backpack.extensions import BatchGrad
import torch
from domainbed.lib.diversity.standard_diversity import DiversityLoss
from domainbed.lib.diversity import kernel_similarity


def compute_kernel(classifiers, loss, center_gradients=False, normalize_gradients=False):
    with backpack(BatchGrad()):
        loss.backward(inputs=list(classifiers.parameters()), retain_graph=True, create_graph=True)
    fmaps = [torch.cat(
            [p.grad_batch.view(p.grad_batch.shape[0], -1) for p in classifier.parameters()], dim=1
        ) for classifier in classifiers]
    fmaps = torch.stack(fmaps, dim=0)  # num_members, bsize, num_weights

    if center_gradients:
        # Center around the weights dimension
        fmaps = fmaps - fmaps.mean(dim=2, keepdim=True)
    if normalize_gradients:
        # Normalize around the weights dimension
        fmaps = fmaps / fmaps.norm(dim=2, keepdim=True)
    kernels = torch.einsum("nbg,ndg->nbd", fmaps, fmaps)
    return kernels  # num_members, bsize, bsize


class L2KernelDistance(DiversityLoss):

    def similarity(self, kernels):
        return kernel_similarity.batch_l2(kernels)

    def forward(self, nlls_per_member, classifiers, **kwargs):
        """Diversity in kernel distance
        Args:
            logits : tensor of shape (num_members, num_domains)
        """
        kernels = compute_kernel(classifiers, nlls_per_member.mean(dim=1).sum())
        loss = self.similarity(kernels)

        return {"loss_div": loss}


class CosKernelDistance(L2KernelDistance):

    def similarity(self, kernels):
        return kernel_similarity.batch_cos(kernels)


class DotKernelDistance(L2KernelDistance):

    def similarity(self, kernels):
        return kernel_similarity.batch_dot(kernels)


class L1KernelDistance(L2KernelDistance):

    def similarity(self, kernels):
        return kernel_similarity.batch_l1(kernels)
