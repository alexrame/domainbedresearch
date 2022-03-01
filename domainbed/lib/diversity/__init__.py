from domainbed.lib.diversity import (
    kernel_diversity, ib_diversity, standard_diversity, sampling_diversity)

def cst(**kwargs):
    return None

DICT_NAME_TO_DIVERSIFIER = {
    "none": cst,
    # gradients
    "L2KernelDistance": kernel_diversity.L2KernelDistance,
    # features
    "IBDiversity": ib_diversity.IBDiversity,
    # predictions
    "LogitDistance": standard_diversity.LogitDistance,
    "KLPreds": standard_diversity.KLPreds,
    "CEDistance": standard_diversity.CEDistance,
    "ADP": standard_diversity.ADP,
    # data
    "GroupDRO": sampling_diversity.GroupDRO,
    "Bagging": sampling_diversity.Bagging,
    "BaggingPerDomain": sampling_diversity.BaggingPerDomain,
}
