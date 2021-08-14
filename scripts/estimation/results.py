from typing import Iterator
from estimation.fit import EllipseModel, PlaneModelND, HomographyModel
from estimation.utils import get_residuals
from test_configuration.utils import _read_yaml

import numpy as np
import sys, os

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

if __name__ == '__main__':

    print(f'[*] Generating results ...')

    model_class = str(sys.argv[1])
    batch_id = str(sys.argv[2])

    # read batch params
    current_path = os.path.dirname(os.path.realpath(__file__))
    scripts_path = current_path[:-11]
    yaml_file = f'{scripts_path}/test_configuration/params/batch_group.yaml'
    batch_group_params = _read_yaml(yaml_file)
    save_path = batch_group_params['group_params']['save_path']
    estimators_names = batch_group_params['group_params']['estimators_names']
	
    batch_params = _read_yaml(f'{save_path}/{model_class}/{batch_id}/batch_params.yaml')
    save_path = batch_params['save_path']
    n_tests = batch_params['n_tests']
    model = eval(batch_params['model_params']['model_class'])()
    estimators_names = batch_params['estimators_names']
    n_estimators = len(estimators_names)
    
    # counter of estimators whose tests results have been generated
    finished_estimators = 1

    for estimator in estimators_names:
    
        # counter tests whose results have been generated
        finished_tests = 0

        n_successful_tests = n_tests # at first we assume all tests have been successfully estimated

        # Errors lists
        estimation_errors = [] # estimation errors, depends on model_class
        abs_errors = []     # absolute and relative errors for each component of the parameters vector
        rel_errors = []	
        iterations_list = []     # iterations of the iterative reestimation stage
            
        for test_id in range(n_tests):

            try:
                # model original and estimated params
                test_params = _read_yaml(f'{save_path}/tests_params/test_{test_id}.yaml')
                params_original = test_params['model_params']['params']
                params_estimated = np.loadtxt(f'{save_path}/results/{estimator}/test_{test_id}_params.txt')
                iterations = np.loadtxt(f'{save_path}/results/{estimator}/test_{test_id}_iterations.txt')

                if model.__class__ == HomographyModel:
                    # read dataset and original inliers to compute RMSE
                    data1 = np.loadtxt(f'{save_path}/datasets/test_{test_id}_proj1.txt')
                    data2 = np.loadtxt(f'{save_path}/datasets/test_{test_id}_proj2.txt')
                    data = np.column_stack((data1,data2))
                    original_inliers = np.loadtxt(f'{save_path}/datasets/test_{test_id}_inliers.txt').astype(bool)

            except IOError:
                n_successful_tests -= 1
                print(f'Test {str(id + 1)} is not considered in results')
                continue
            except ValueError:
                n_successful_tests -= 1
                print(f'Cannot read ../{estimator}/test_{id}_params.txt')
                continue

            # convert params to general coefficients and compute estimation_error
            if model.__class__ == PlaneModelND:
                params_original = np.asarray(params_original).reshape(2, -1)
                estimation_error, is_antiparallel = _angle_between_vectors( params_original[1], 
                                                                            params_estimated[1])
                if is_antiparallel:
                    params_estimated[1] = -params_estimated[1]

            params_original = model.get_general_params(params_original)
            params_estimated = model.get_general_params(params_estimated)

            if model.__class__ == HomographyModel:
                estimation_error = _compute_RMSE(params_original, params_estimated, data, original_inliers)
            
            estimation_errors = np.append(estimation_errors, estimation_error)

            # abs and rel errors
            abs_error, rel_error = _get_estimation_error(params_original, params_estimated)
            abs_errors = np.append(abs_errors, abs_error)
            rel_errors = np.append(rel_errors, rel_error)
            # some additional info
            iterations_list = np.append(iterations_list, iterations)
            
            finished_tests += 1

            percentage_completed = "%.2f" % float(finished_tests / n_tests * 100)
            sys.stdout.write(f'\rGenerating results for {batch_id} with {n_tests} tests | {finished_estimators}/{n_estimators} Completed | {percentage_completed} % Complete')
            sys.stdout.flush()
    
        # reshape and save
        estimation_errors = estimation_errors.reshape(n_successful_tests, -1)
        abs_errors = abs_errors.reshape(n_successful_tests, -1)
        rel_errors = rel_errors.reshape(n_successful_tests, -1)
        iterations_list = iterations_list.reshape(n_successful_tests, -1)

        np.savetxt(f'{save_path}/results/{estimator}/00_estimation_errors.txt', estimation_errors)
        np.savetxt(f'{save_path}/results/{estimator}/00_abs_errors.txt', abs_errors)
        np.savetxt(f'{save_path}/results/{estimator}/00_rel_errors.txt', rel_errors)
        np.savetxt(f'{save_path}/results/{estimator}/00_iterations.txt', iterations_list)

        finished_estimators += 1
    
    print(' ')