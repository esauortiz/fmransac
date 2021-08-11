import numpy as np
import sys
import yaml
import io

from estimation.myfit import CircleModel, LineModelND, EllipseModel, PlaneModelND, BaseModel, HomographyModel, get_residuals

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
	abs_error = params_original - params_estimated
	rel_error = abs_error / params_original
	rel_error[rel_error == np.inf] = 0
	rel_error[rel_error == -np.inf] = 0
	return abs_error, rel_error

def check_antiparallelism(original_normal_vector, estimated_normal_vector):
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

def get_hplane_params(params):
	params = np.asarray(params).reshape(2, -1)
	origin, normal_vector = params
	normal_vector /= np.linalg.norm(normal_vector)
	x_n = np.dot(np.transpose(normal_vector), origin)
	return np.append(normal_vector, x_n)

def get_line_params(params):

	origin = params[0]
	direction = params[1]
	
	#lambda parace ser palabra reservada
	landa = 100000

	pos_point = origin + landa * direction
	neg_point = origin - landa * direction

	data = np.array([pos_point, neg_point])
	_,_, v = np.linalg.svd(data, full_matrices=False)
	coeffs = v[1]
	
	k = origin / direction
	origin = origin - k[0]*(direction)
	d = np.dot(coeffs, origin)
	return np.array([*coeffs, -d])

def compute_theta(original_normal_vector, estimated_normal_vector):
	"""
	Parameters
	----------
	original_normal_vector : (N, ) array
	estimated_normal_vector : (N, ) array
	Returns
	-------
	theta : float
		Angle between vectors
	is_antiparallel:
		True if both vectors have the same direction
	"""
	onv = original_normal_vector / np.linalg.norm(original_normal_vector)
	env = estimated_normal_vector / np.linalg.norm(estimated_normal_vector)
	same_dir_theta = np.rad2deg(np.arccos(np.dot(env,onv)/(np.linalg.norm(env)*np.linalg.norm(onv))))
	antiparallel_dir_theta = np.rad2deg(np.arccos(np.dot(-env,onv)/(np.linalg.norm(-env)*np.linalg.norm(onv))))
	if same_dir_theta < antiparallel_dir_theta:
		return same_dir_theta, False
	else:
		return antiparallel_dir_theta, True

def compute_eucliden_distance(params_original, params_estimated):
	result = np.sum((params_original - params_estimated)**2)
	return np.sqrt(result)

def get_circle_params(params):
	xc, yc, r = params
	return np.array([-2*xc, -2*yc, ((xc ** 2) + (yc ** 2) - (r ** 2))])

def get_homography_params(params):

	s_cos_phi, s_sin_phi, tx, ty = params
	"""
	H = np.array([[s_cos_phi, -s_sin_phi, tx],
                  [s_sin_phi, s_cos_phi,  ty],
                  [0,         0,          1]])
	H = np.linalg.inv(H)
	H /= H[2,2]
	s_cos_phi = H[0,0]
	s_sin_phi = H[1,0]
	tx = H[0,2]
	ty = H[1,2]
	"""
	s = np.sqrt(s_cos_phi**2 + s_sin_phi**2)
	phi = np.arctan(s_sin_phi/s_cos_phi)
	params = np.array([s, phi, tx, ty])
	return params

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

def compute_RMSE(params_original_i, params_estimated, data, inliers):
	# since tp points are contaminated with gaussian noise it will be removed, then a good estimation will have RMSE \approx 0
	residuals_original = get_residuals(data, HomographyModel, params_original_i)
	residuals_estimated = get_residuals(data, HomographyModel, params_estimated)
	squared_error = (residuals_original - residuals_estimated)**2
	RMSE = np.sqrt(np.mean(squared_error[inliers]))
	return RMSE

if __name__ == '__main__':
	# Read test params
	save_path = sys.argv[1]
	test_num = int(sys.argv[2])
	model_class =  sys.argv[3]
	model_estimator = list(sys.argv[4].split(','))
	is_HomographyModel = False
	
	# Original model params
	params_original =[]
	# NSE_denominator will be computed first i.e. Err(d_i; M*) ** 2
	NSE_denominators = []

	if model_class == 'LineModelND':
		model_class = LineModelND
		_get_params = get_line_params
	elif model_class == 'CircleModel':
		model_class = CircleModel
		_get_params = get_circle_params
	elif model_class == 'EllipseModel':
		model_class = EllipseModel
		_get_params = get_ellipse_params
	elif model_class == 'PlaneModelND':
		model_class = PlaneModelND
		_get_params = get_hplane_params
	elif model_class == 'HomographyModel':
		model_class = HomographyModel
		_get_params = get_homography_params
		is_HomographyModel = True

	results_len = test_num

	for id in range(test_num):
		yaml_file = save_path + '/yaml/test_' + str(id + 1) + '.yaml'
		with open(yaml_file, 'r') as stream:
			test_params = yaml.safe_load(stream)
		params_original_i = test_params['data_params']['model_params']
		params_original = np.append(params_original, params_original_i)
		try:
			# Try to read data
			if is_HomographyModel:
				data1 = np.loadtxt(save_path + '/data/test_' + str(id + 1) + '_proj1.txt')
				data2 = np.loadtxt(save_path + '/data/test_' + str(id + 1) + '_proj2.txt')
				data = np.column_stack((data1,data2))
			else:
				data = np.loadtxt(save_path + '/data/test_' + str(id + 1) + '.txt')

			# Try to read inlier mask
			inliers = np.loadtxt(save_path + '/data/test_' + str(id + 1) + '_inliers.txt').astype(bool)
		except IOError:
			results_len -= 1
			print(f'File `.../data/test_{id+1}_inliers.txt` not found')
			NSE_denominators = np.append(NSE_denominators, np.nan)
			continue
		
		residuals_original = get_residuals(data, model_class, params_original_i)
		NSE_denominators = np.append(NSE_denominators, np.sum(residuals_original[inliers] ** 2))

	params_original = params_original.reshape(test_num, -1)
	NSE_denominators = NSE_denominators.reshape(test_num, -1)

	for model_e in model_estimator:
		print(f'Generating results for {model_e}...')
		
		# Erros arrays
		abs_errors = []
		rel_errors = []	
		params_origi = []
		params_estim = []
		more_info = []
		NSE_values = []
		theta_values = []
		RMSE_values = []
		#euclidean_d_values = []
		linfinity_values = []

		results_len = test_num
		for id in range(test_num):

			try:
				# model estimated params
				params_estimated = np.loadtxt(save_path + '/results/' + model_e + '/test_' + str(id + 1) + '_params.txt')
				more_info_i = np.loadtxt(save_path + '/results/' + model_e + '/test_' + str(id + 1) + '_more_info.txt')
				if is_HomographyModel:
					data1 = np.loadtxt(save_path + '/data/test_' + str(id + 1) + '_proj1.txt')
					data2 = np.loadtxt(save_path + '/data/test_' + str(id + 1) + '_proj2.txt')
					data = np.column_stack((data1,data2))
					estimated_inliers = np.loadtxt(save_path + '/results/' + model_e + '/test_' + str(id + 1) + '_inliers.txt')
				else:
					data = np.loadtxt(save_path + '/data/test_' + str(id + 1) + '.txt')
				#original inliers
				inliers = np.loadtxt(save_path + '/data/test_' + str(id + 1) + '_inliers.txt').astype(bool)
			except IOError:
				results_len -= 1
				print(f'Test {str(id + 1)} is not considered in results')
				print("IOError")
				continue
			except ValueError:
				results_len -= 1
				print(f'Cannot read .../{model_e}/test_{id}_params.txt')
				continue
		
			# model original params
			params_original_i = params_original[id]
			# NSE
			NSE_value_i = _compute_NSE(data, model_class, NSE_denominators[id], params_estimated, inliers)
			NSE_values = np.append(NSE_values, NSE_value_i)
		
		 	# convert params to general coefficients
			if model_class == PlaneModelND:
				params_original_i = params_original_i.reshape(2, -1)
				#if check_antiparallelism(params_original_i[1], params_estimated[1]) == True:
				#	params_estimated[1] = -params_estimated[1]
				theta_i, is_antiparallel = compute_theta(params_original_i[1], params_estimated[1])
				if is_antiparallel:
					params_estimated[1] = -params_estimated[1]

			params_original_i = _get_params(params_original_i)
			params_estimated = _get_params(params_estimated)

			if model_class == HomographyModel or model_class == EllipseModel:
				theta_i, _ = compute_theta(params_original_i, params_estimated)
				RMSE_i = compute_RMSE(params_original_i, params_estimated, data, inliers)
				RMSE_values = np.append(RMSE_values, RMSE_i)
				#euclidean_d_i = compute_eucliden_distance(params_original_i, params_estimated)
			
			theta_values = np.append(theta_values, theta_i)
			#euclidean_d_values = np.append(euclidean_d_values, euclidean_d_i)

			# compute and save results
				# abs and rel errors
			abs_error_i, rel_error_i = _get_estimation_error(params_original_i, params_estimated)
			abs_errors = np.append(abs_errors, abs_error_i)
			rel_errors = np.append(rel_errors, rel_error_i)
				# saving original and estimated params
			params_origi = np.append(params_origi, params_original_i)
			params_estim = np.append(params_estim, params_estimated)
				# some aditional info
			more_info = np.append(more_info, more_info_i)

			# linfinity
			linfinity_values = np.append(linfinity_values, np.max(rel_error_i))

		# reshape and save
		abs_errors = abs_errors.reshape(results_len, -1)
		rel_errors = rel_errors.reshape(results_len, -1)
		#params_origi = params_origi.reshape(results_len, -1)
		#params_estim = params_estim.reshape(results_len, -1)
		more_info = more_info.reshape(results_len, -1)
		NSE_values = NSE_values.reshape(results_len, -1)
		RMSE_values = RMSE_values.reshape(results_len, -1)
		theta_values = theta_values.reshape(results_len, -1)
		#euclidean_d_values = euclidean_d_values.reshape(results_len, -1)
		linfinity_values = linfinity_values.reshape(results_len, -1)
		
		np.savetxt((save_path + '/results/' + model_e + '/00_abs_errors.txt'), abs_errors)
		np.savetxt((save_path + '/results/' + model_e + '/00_rel_errors.txt'), rel_errors)
		np.savetxt((save_path + '/results/' + model_e + '/00_NSE_values.txt'), NSE_values)
		np.savetxt((save_path + '/results/' + model_e + '/00_RMSE_values.txt'), RMSE_values)
		np.savetxt((save_path + '/results/' + model_e + '/00_linfinity_values.txt'), linfinity_values)
		#np.savetxt((save_path + '/results/' + model_e + '/00_params_origi.txt'), params_origi)
		#np.savetxt((save_path + '/results/' + model_e + '/00_params_estim.txt'), params_estim)
		np.savetxt((save_path + '/results/' + model_e + '/00_more_info.txt'), more_info)
		np.savetxt((save_path + '/results/' + model_e + '/00_theta_values.txt'), theta_values)
		