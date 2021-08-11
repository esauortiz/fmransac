import numpy as np

def _is_inlier(data, model, residual_threshold):
    data = np.array(data.reshape(1,-1))
    data_residuals = np.abs(model.residuals(data))
    return data_residuals < residual_threshold

def get_residuals(data, model_class, model_params):
    data_model = model_class()
    data_residuals = np.abs(data_model.residuals(data, model_params))
    return data_residuals   

def _check_data_dim(data, dim):
    if data.ndim != 2 or data.shape[1] != dim:
        raise ValueError('Input data must have shape (N, %d).' % dim)

def _check_data_atleast_2D(data):
    if data.ndim < 2 or data.shape[1] < 2:
        raise ValueError(f'Input data must be at least 2D. Data shape is {data.shape}')

def _norm_along_axis(x, axis):
    """NumPy < 1.8 does not support the `axis` argument for `np.linalg.norm`."""
    return np.sqrt(np.einsum('ij,ij->i', x, x))

def _musigma_norm(data):
    """Normalizes data based on
    mu-sigma normalization
    Parameters
    ----------
    data : (N, dim) array
        N points in a space of dimensionality dim >= 2.
    Returns
    -------
    data : (N, dim) array
        Normalized data
    H : (dim + 1, dim + 1) array
        Normalization matrix
    """
    dim = data.shape[1]

    # vector of means of every column in data
    mu_ax = data.mean(axis = 0)
    # vector of std of every column in data
    std_ax = data.std(axis = 0)

    # building normalization matrix H
    diag = np.ones((dim + 1,))
    diag[0:dim,] = 1 / std_ax
    H = np.diag(diag)
    H[:, dim][0:dim,] = -mu_ax/std_ax
    
    """ example of H matrix of 2D data
    
    H = np.array([  [1/s_x, 0,      -mu_x/s_x],
                    [0,     1/s_y,  -mu_y/s_y],
                    [0,     0,      1]])
    """

    # apply normalization and return
    data_matrix = np.ones((data.shape[0],data.shape[1] + 1))
    data_matrix[:,:-1] = data
    data =  np.dot(H, data_matrix.T).T[:,0:dim]
    return data, H

def _maxmin_norm(data):
    """Normalizes data based on
    mu-sigma normalization
    Parameters
    ----------
    data : (N, dim) array
        N points in a space of dimensionality dim >= 2.
    Returns
    -------
    data : (N, dim) array
        Normalized data
    H : (dim + 1, dim + 1) array
        Normalization matrix
    """
    dim = data.shape[1]

    # vector of max of every column in data
    max_ax = data.max(axis = 0)
    # vector of min of every column in data
    min_ax = data.min(axis = 0)
    # vector of r of every column in data
    r_ax = max_ax - min_ax

    # building normalization matrix H
    diag = np.ones((dim + 1,))
    diag[0:dim,] = 1 / r_ax
    H = np.diag(diag)
    H[:, dim][0:dim,] = -min_ax/r_ax
    
    """ example of H matrix of 2D data
    
    H = np.array([  [1/r_x, 0,      -min_x/r_x],
                    [0,     1/r_y,  -min_y/r_y],
                    [0,     0,      1]])
    """

    # apply normalization and return
    data_matrix = np.ones((data.shape[0],data.shape[1] + 1))
    data_matrix[:,:-1] = data
    data =  np.dot(H, data_matrix.T).T[:,0:dim]
    return data, H
