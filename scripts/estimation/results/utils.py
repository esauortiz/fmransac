import numpy as np

def get_metric(values, stat_type):
    """ Returns metric given an (N,) array
        and a statistical value type as string
    Parameters
    ----------
    values : (N, ) array
        N values from which the statistical value is calculated
    stat_type : string
        statistical value type requested
    Returns
    -------
    metric : float
        statistical value
    """

    if stat_type[0] == 'P':
        percentile = stat_type[1:]
        return np.percentile(values, percentile)
    elif stat_type == 'mean':
        return np.mean(values)
    elif stat_type == 'sd':
        return np.std(values)