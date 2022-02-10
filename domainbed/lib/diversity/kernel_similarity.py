import torch
import torch.nn.functional as F
from backpack import backpack
from backpack.extensions import BatchGrad


def compute_kernels(loss, classifiers, bsize, detach_first=False, center_gradients="none", all_y=None, normalize_gradients=False):
    with backpack(BatchGrad()):
        loss.backward(retain_graph=True, create_graph=True, inputs=[p for c in classifiers for p in c.parameters()])
    fmaps = [torch.cat([p.grad_batch.view(bsize, -1) for p in classifier.parameters()], dim=1) for classifier in classifiers]   # (bsize, num_params)
    if detach_first:
        fmaps[0] = fmaps[0].detach()
    if center_gradients == "all":
        fmaps = [fmap - fmap.mean(dim=1, keepdim=True) for fmap in fmaps]
    elif center_gradients == "classes":
        assert all_y is not None
        fmaps = torch.stack(fmaps)  # num_classif, bsize, dim (num_weights)
        for val in torch.unique(all_y):
            indexes = (all_y==val).nonzero().squeeze()
            mean = fmaps[:, indexes].mean(dim=2, keepdim=True)
            fmaps[:, indexes] = fmaps[:, indexes]  - mean
    
    if normalize_gradients:
        fmaps = [fmap / fmap.norm(dim=1, keepdim=True) for fmap in fmaps]

    kernels = [fmap @ fmap.t() for fmap in fmaps]
    return kernels


def dot(kernelA, kernelB):
    """dot product between kernels

    Args:
        kernelA: (N, N) tensor
        kernelB (N, N) tensor

    """
    return (kernelA * kernelB).sum()

def cos(kernelA, kernelB):
    similarity = dot(kernelA, kernelB)
    return (similarity / kernelA.norm() / kernelB.norm())

def center(kernel):
    return kernel - torch.mean(kernel, 0)

def center_dot(kernelA, kernelB):
    kernelA = center(kernelA)
    kernelB = center(kernelB)
    return dot(kernelA, kernelB)

def center_cos(kernelA, kernelB):
    kernelA = center(kernelA)
    kernelB = center(kernelB)
    return cos(kernelA, kernelB)


def batch_dot(kernels):
    """batch dot product

    Args:
        kernels (tensor): (num_kernels, dim1, dim2)
    returns:
        similarity_matrix (num_kernels, num_kernels)
    """
    similarity_matrix = torch.einsum("nab,mab->nm", kernels, kernels)  # num_kernels, bsize, bsize
    similarities = torch.triu(similarity_matrix, diagonal=1)  # upper left triangle
    return torch.mean(similarities)

def batch_center_dot(kernels):
    """
    kernels:  (num_kernels, bsize, bsize)
    """
    kernels = kernels - torch.mean(kernels, dim=1, keepdim=True)
    return batch_dot(kernels)

def batch_cos(kernels):
    """
    kernels: (num_kernels, bsize, bsize)
    """
    # normalize kernels
    norm_kernels = kernels / torch.norm(kernels, dim=[1, 2], keepdim=True) 
    similarity_matrix = torch.einsum("nab,mab->nm", norm_kernels, norm_kernels)  # num_kernels, bsize, bsize

    sim = similarity_matrix.mean()
    if sim.isnan():
        breakpoint()
    return sim

# similarities = torch.triu(similarity_matrix, diagonal=1)  # upper right triangle, (num_kernels, num_kernels)
# norms = torch.diagonal(similarity_matrix)   # (num_kernels,)
# similarities = similarity_matrix - torch.diag(torch.diag(similarity_matrix))

def batch_center_cos(kernels):
    kernels = kernels - torch.mean(kernels, dim=1, keepdim=True)
    return batch_cos(kernels)

def batch_l2(kernels):
    """
    Args:
        kernels: (num_kernels, bsize, bsize)
    """
    n = kernels.shape[0]
    kernels = kernels.view(n, -1)  # (n, bsize*bsize)
    distances = torch.cdist(kernels, kernels)
    distances = distances - torch.diag(torch.diag(distances))
    return - distances.mean()  # (num_kernels, num_kernels)

def batch_l1(kernels):
    """
    Args:
        kernels: (num_kernels, bsize, bsize)
    """
    n = kernels.shape[0]
    kernels = kernels.view(n, -1)  # (n, bsize*bsize)
    distances = torch.cdist(kernels, kernels, p=1)
    distances = distances - torch.diag(torch.diag(distances))
    return - distances.mean()  # (num_kernels, num_kernels)

def teney_similarity():
    pass


# def linear_cka(fmaps):
#     """
#     fmaps: (num_fmaps, bsize, dim)
#     """
#     # kna,lma->kl
#     # kna,lnb->klab
#     torch.einsum("nbd,mbd->nm")


def batch_similarity(kernels, name, result="none"):
    if name == "dot":
        ks = torch.stack(kernels, dim=0)
        similarity = batch_dot(ks)
    if name == "dot-diagonal":
        kernels = torch.stack([k - torch.diagonal(k) for k in kernels], dim=0)
        similarity = batch_dot(kernels)
    elif name == "center-dot":
        ks = torch.stack(kernels, dim=0)
        similarity = batch_center_dot(ks)
    elif name == "cos":
        ks = torch.stack(kernels, dim=0)
        similarity = batch_cos(ks)
    elif name == "center-cos":
        ks = torch.stack(kernels, dim=0)
        similarity = batch_center_cos(ks)
    elif name == "l2":
        ks = torch.stack(kernels, dim=0)
        similarity = batch_l2(ks)
    elif name == "l1":
        ks = torch.stack(kernels, dim=0)
        similarity = batch_l1(ks)

    # results
    if result == "square":
        similarity = similarity**2
    elif result == "none":
        pass
    elif result == "abs":
        similarity = similarity.abs()
    elif result == "relu":
        similarity = F.relu(similarity)
    else:
        raise ValueError(result)

    return similarity
