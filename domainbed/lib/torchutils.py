import torch

def my_cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    return res.clamp_min_(1e-30)


def gaussian_kernel(x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
    D = my_cdist(x, y)
    K = torch.zeros_like(D)

    for g in gamma:
        K.add_(torch.exp(D.mul(-g)))

    return K


def standard_mmd(x, y, kernel_type):
    if kernel_type == "gaussian":
        Kxx = gaussian_kernel(x, x).mean()
        Kyy = gaussian_kernel(y, y).mean()
        Kxy = gaussian_kernel(x, y).mean()
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

def mmd_ema(x, y, weights_x=None, weights_y=None, list_methods=None):

    if "gaussian" in list_methods:
        Kxx = gaussian_kernel(x, x).mean()
        Kyy = gaussian_kernel(y, y).mean()
        Kxy = gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    mean_x = x.mean(0, keepdim=True)
    mean_y = y.mean(0, keepdim=True)

    transf_x = x
    transf_y = y
    if "notcentered" not in list_methods:
        transf_x = transf_x - mean_x
        transf_y = transf_y - mean_y

    if "weight" in list_methods:
        assert weights_x is not None
        transf_x = transf_x * weights_x.view((weights_x.size(0), 1))
        transf_y = transf_y * weights_y.view((weights_y.size(0), 1))

    if "offdiagonal" in list_methods:
        # cova_x = (transf_x.t() @ transf_x) / (transf_x.size(0) * transf_x.size(1))
        # cova_y = (transf_y.t() @ transf_y) / (transf_y.size(0) * transf_y.size(1))
        cova_x = torch.einsum("na,nb->ab", transf_x,
                                transf_x) / (transf_x.size(0) * transf_x.size(1))
        cova_y = torch.einsum("na,nb->ab", transf_y,
                                transf_y) / (transf_y.size(0) * transf_y.size(1))
    else:
        cova_x = (transf_x).pow(2).mean(dim=0)
        cova_y = (transf_y).pow(2).mean(dim=0)

    mean_diff = (mean_x - mean_y).pow(2).mean()
    cova_diff = (cova_x - cova_y).pow(2).mean()

    if "mean" in list_methods:
        return cova_diff + mean_diff
    return cova_diff


def mmd(x, y, weights_x=None, weights_y=None, list_methods=None):

    if "gaussian" in list_methods:
        Kxx = gaussian_kernel(x, x).mean()
        Kyy = gaussian_kernel(y, y).mean()
        Kxy = gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    mean_x = x.mean(0, keepdim=True)
    mean_y = y.mean(0, keepdim=True)

    transf_x = x
    transf_y = y
    if "notcentered" not in list_methods:
        transf_x = transf_x - mean_x
        transf_y = transf_y - mean_y

    if "weight" in list_methods:
        assert weights_x is not None
        transf_x = transf_x * weights_x.view((weights_x.size(0), 1))
        transf_y = transf_y * weights_y.view((weights_y.size(0), 1))

    if "offdiagonal" in list_methods:
        # cova_x = (transf_x.t() @ transf_x) / (transf_x.size(0) * transf_x.size(1))
        # cova_y = (transf_y.t() @ transf_y) / (transf_y.size(0) * transf_y.size(1))
        cova_x = torch.einsum("na,nb->ab", transf_x,
                                transf_x) / (transf_x.size(0) * transf_x.size(1))
        cova_y = torch.einsum("na,nb->ab", transf_y,
                                transf_y) / (transf_y.size(0) * transf_y.size(1))
    else:
        cova_x = (transf_x).pow(2).mean(dim=0)
        cova_y = (transf_y).pow(2).mean(dim=0)

    mean_diff = (mean_x - mean_y).pow(2).mean()
    cova_diff = (cova_x - cova_y).pow(2).mean()

    if "mean" in list_methods:
        return cova_diff + mean_diff
    return cova_diff
