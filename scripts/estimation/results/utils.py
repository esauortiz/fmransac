from estimation.fit import HomographyModel
from estimation.utils import get_residuals

import numpy as np

# results main utils
def _get_estimation_error(params_original, params_estimated):
    """Computes absolute and relative error
    Parameters
    ----------
    params_original : (N, ) array
        N original parameters
    params_estimated : (N, ) array
        N estimated parameters
    Returns
    -------
    abs_error: (N, ) array
        absolute errors
    rel_error: (N, ) array
        relative errors
    """
    abs_error = []
    rel_error = []
    for param_original, param_estimated in zip(params_original, params_estimated):
        abs_error.append(param_original - param_estimated)
        if param_original != 0: # angle division by 0
            rel_error.append((param_original - param_estimated) / param_original)
        else:
            rel_error.append(0)
    return abs_error, rel_error

def _check_antiparallelism(original_normal_vector, estimated_normal_vector):
  """Checks if the normal vectors contained in params are antiparallel
    or near antiparallelism i.e. dot(a, b)/(a.magnitude*b.magnitude) < 0
  Parameters
  ----------
  params_original : (N, ) array
      N original parameters
  params_estimated : (N, ) array
      N estimated parameters
  Returns
  -------
  bool :
    indicating if the vectors are antiparallel
  """
  onv = original_normal_vector / np.linalg.norm(original_normal_vector)
  env = estimated_normal_vector / np.linalg.norm(estimated_normal_vector)
  # a · b = |a| × |b| × cos(θ)
  # To get anti-parallel you want θ = τ/2 and so cos(τ/2) = -1
  # dot(a, b)/(a.magnitude*b.magnitude) == -1
  # condition has been changed to '< 0' because the vectors could not be strictly antiparallel
  return np.dot(env,onv)/(np.linalg.norm(env)*np.linalg.norm(onv)) < 0

def _angle_between_vectors(original_normal_vector, estimated_normal_vector):
    """
    Parameters
    ----------
    original_normal_vector : (N, ) array
    estimated_normal_vector : (N, ) array
    Returns
    -------
    angle : float
        Angle between vectors
    is_antiparallel:
        True if both vectors have the same direction
    """
    onv = original_normal_vector / np.linalg.norm(original_normal_vector)
    env = estimated_normal_vector / np.linalg.norm(estimated_normal_vector)
    same_dir_angle = np.rad2deg(np.arccos(np.dot(env,onv)/(np.linalg.norm(env)*np.linalg.norm(onv))))
    antiparallel_dir_angle = np.rad2deg(np.arccos(np.dot(-env,onv)/(np.linalg.norm(-env)*np.linalg.norm(onv))))

    if same_dir_angle < antiparallel_dir_angle:
        return same_dir_angle, False
    else:
        return antiparallel_dir_angle, True

def _compute_NSE(data, model_class, NSE_denominator, params_estimated, inliers):
	"""Compute the 'Normalized squared error of inliers' (NSE).
		NSE comes from Choi and Kim’ problem definition [4]. NSE is close to 1 when the magnitude 
		of error by sthe estimated line is near the magnitude of error by the truth.
		[4] Sunglok Choi and Jong-Hwan Kim. Robust regression to varying data distribution and
		its application to landmark-based localization. In Proceedings of the IEEE Conference
		on Systems, Man, and Cybernetics, October 2008.
	Parameters
	----------
	data : (N, dim) array
		data with N 'dim' dimensional points
	model_class : model
		model to which the data belongs
	NSE_numerator : float
	    NSE numerator i.e. Err(d_i; M*) ** 2 (d_i belongs to D_in)
	params_estimated : (n, ) array
		estimated model parameters
	inliers : (N, ) array
		original model inliers i.e. D_in
	Returns
	-------
	NSE : float
		Normalized squared error of inliers
	"""
	residuals_estimated = get_residuals(data, model_class, params_estimated)
	NSE_numerator = np.sum(residuals_estimated[inliers] ** 2)
	
	return NSE_numerator / NSE_denominator

def _compute_RMSE(params_original, params_estimated, data, inliers):
	# since tp points are contaminated with gaussian noise it will be removed, 
    # then a good estimation will have RMSE \approx 0
	residuals_original = get_residuals(data, HomographyModel, params_original)
	residuals_estimated = get_residuals(data, HomographyModel, params_estimated)
	squared_error = (residuals_original - residuals_estimated)**2
	RMSE = np.sqrt(np.mean(squared_error[inliers]))
	return RMSE

# tabulate utils
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
        percentile = int(stat_type[1:])
        return np.percentile(values, percentile)
    elif stat_type == 'mean':
        return np.mean(values)
    elif stat_type == 'std':
        return np.std(values)

def get_row_labels(batch_group_params):
    """ Given a dictionary of params will return which parameter
        has a different value along all the batches in the batch
        group
    Parameters
    ----------
    batch_group_params : dictionary
        parameters of the batch group
    Returns
    -------
    column_labels: (n_batches + 1, ) array
    """
    dataset_params = batch_group_params['dataset_params']
    ransac_params = batch_group_params['ransac_params']
    params = {**dataset_params, **ransac_params} # merge two parameters dictionaries
    column_labels = [None]

    for key in params: # search which parameter's values varies along batches
        if len(params[key]) > 1: column_labels[0] = key

    # append parameter's values
    for value in params[column_labels[0]]: column_labels.append(value)    

    return column_labels