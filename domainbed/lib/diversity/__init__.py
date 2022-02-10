from domainbed.lib.diversity import (
    kernel_diversity, standard_diversity, ib_diversity)

def cst(**kwargs):
    return None

DICT_NAME_TO_DIVERSIFIER = {
    "none": cst,
    "L2KernelDistance": kernel_diversity.L2KernelDistance,
    "IBDiversity": ib_diversity.IBDiversity,
    "LogitDistance": standard_diversity.LogitDistance,
    "CEDistance": standard_diversity.CEDistance,
    "GroupDRO": standard_diversity.GroupDRO,
    "Bagging": standard_diversity.Bagging,
    "BaggingPerDomain": standard_diversity.BaggingPerDomain,
}
