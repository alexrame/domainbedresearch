def _process_predictions(y, y_pred1, y_pred2):
    size_y = len(y)
    if size_y != len(y_pred1) or size_y != len(y_pred2):
        raise ValueError('The vector with class labels must have the same size.')

    correct_1 = (y_pred1 == y)
    correct_2 = (y_pred2 == y)
    N11 = (correct_1 & correct_2).sum()
    N01 = ((~correct_1) & correct_2).sum()
    N10 = (correct_1 & (~correct_2)).sum()
    N00 = size_y - N11 - N01 - N10

    return N00 / size_y, N10 / size_y, N01 / size_y, N11 / size_y


def Q_statistic(y, y_pred1, y_pred2):
    """Calculates the Q-statistics diversity measure between a pair of
    classifiers. The Q value is in a range [-1, 1]. Classifiers that tend to
    classify the same object correctly will have positive values of Q, and
    Q = 0 for two independent classifiers.
    Parameters
    ----------
    y : array of shape = [n_samples]:
        class labels of each sample.
    y_pred1 : array of shape = [n_samples]:
              predicted class labels by the classifier 1 for each sample.
    y_pred2 : array of shape = [n_samples]:
              predicted class labels by the classifier 2 for each sample.
    Returns
    -------
    Q : The q-statistic measure between two classifiers
    """
    N00, N10, N01, N11 = _process_predictions(y, y_pred1, y_pred2)
    Q = ((N11 * N00) - (N01 * N10)) / ((N11 * N00) + (N01 * N10))
    return Q



def double_fault(y, y_pred1, y_pred2):
    """Calculates the double fault (df) measure. This measure represents the
    probability that both classifiers makes the wrong prediction. A lower value
    of df means the base classifiers are less likely to make the same error.
    This measure must be minimized to increase diversity.
    Parameters
    ----------
    y : array of shape = [n_samples]:
        class labels of each sample.
    y_pred1 : array of shape = [n_samples]:
              predicted class labels by the classifier 1 for each sample.
    y_pred2 : array of shape = [n_samples]:
              predicted class labels by the classifier 2 for each sample.
    Returns
    -------
    df : The double fault measure between two classifiers
    References
    ----------
    Giacinto, Giorgio, and Fabio Roli. "Design of effective neural network
    ensembles for image classification purposes."
    Image and Vision Computing 19.9 (2001): 699-707.
    """
    N00, _, _, _ = _process_predictions(y, y_pred1, y_pred2)
    df = N00
    return df


def single_fault(y, y_pred1, y_pred2):
    _, N10, N01, _ = _process_predictions(y, y_pred1, y_pred2)
    return N01 + N10


def ratio_errors(y, y_pred1, y_pred2):
    """Calculates Ratio of errors diversity measure between a pair of
    classifiers. A higher value means that the base classifiers are less likely
    to make the same errors. The ratio must be maximized for a higher diversity
    Parameters
    ----------
    y : array of shape = [n_samples]:
        class labels of each sample.
    y_pred1 : array of shape = [n_samples]:
              predicted class labels by the classifier 1 for each sample.
    y_pred2 : array of shape = [n_samples]:
              predicted class labels by the classifier 2 for each sample.
    Returns
    -------
    ratio : The q-statistic measure between two classifiers
    References
    ----------
    Aksela, Matti. "Comparison of classifier selection methods for improving
    committee performance."
    Multiple Classifier Systems (2003): 159-159.
    """
    # N00, N10, N01, N11 = _process_predictions(y, y_pred1, y_pred2)
    N00, N10, N01, N11 = _process_predictions(y, y_pred1, y_pred2)
    if N00 == 0:
        print("No shared errors !")
        ratio = 2 * (N01 + N10)
    else:
        ratio = (N01 + N10) / N00
    return ratio


def agreement_measure(y, y_pred1, y_pred2):
    """Calculates the agreement measure between a pair of classifiers. This
    measure is calculated by the frequency that both classifiers either
    obtained the correct or incorrect prediction for any given sample
    Parameters
    ----------
    y : array of shape (n_samples):
        class labels of each sample.
    y_pred1 : array of shape (n_samples):
              predicted class labels by the classifier 1 for each sample.
    y_pred2 : array of shape (n_samples):
              predicted class labels by the classifier 2 for each sample.
    Returns
    -------
    agreement : The frequency at which both classifiers agrees
    """
    N00, _, _, N11 = _process_predictions(y, y_pred1, y_pred2)
    agreement = N00 + N11
    return agreement
