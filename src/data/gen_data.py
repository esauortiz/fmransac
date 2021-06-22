import numpy as np
import sys
import yaml
import io
import random

from estimation.myfit import CircleModel, LineModelND, EllipseModel, PlaneModelND, is_inlier

def _check_dim_match(data, target_dim):
	"""Checks dimension match beetween data and requested target dimensions
	Parameters
	----------
	data : (N, dim) array
	    N points with dim dimensions
	target_dim : tuple
		Indicating the dimensions to which noise is added
	Returns
	-------
	target_dim : tuple
	Raises
	------
	ValueError
	    If there is not a match
	"""
	target_dim = list(set(target_dim))

	if max(target_dim) >= np.shape(data)[1]:
		raise ValueError('The dimension cannot be greater than %d' % (np.shape(data)[1] - 1) )
	else:
		return target_dim

def get_bbox_limits(data, tolerance):
	# 'dim' rows and 2 columns (min, max)
	limits = np.empty((np.shape(data)[1], 2), dtype = float)
	i = 0
	for col in np.transpose(data):
		limits[i] = np.array([min(col) - tolerance, max(col) + tolerance])
		i += 1
	return limits

def is_in_bbox(coord, bbox_limits):
	for bbox_ax_lims, component in zip(bbox_limits, *coord):
		if (component <= bbox_ax_lims[0]) or (component >= bbox_ax_lims[1]):
			return False
	return True
	
def gen_uniform_noise(	data, model_class, model_params, residual_threshold, points_num,
					 	max_trials, noise_level, target_dim, outlier_ratio, seed, bbox_limits = None,
					 	bbox_limits_tolerance = 0):
	"""Adds uniformly distributed noise
	Parameters
	----------
	Returns
	-------
	noisy data : (N, dim) array
	"""
	np.random.seed(None)
	random.seed(None)
	points_num = int(points_num)

	origin = data.mean(axis=0)
	dim = np.size(origin)

	inliers_num = points_num * (1 - outlier_ratio)
	outliers_num = points_num * outlier_ratio

	inliers_counter = 0
	outliers_counter = 0

	target_dim = _check_dim_match(data, target_dim)
	noise_list = np.empty((points_num, dim), dtype = float)
	i = 0
	trials = 0

	if bbox_limits is None:
		bbox_limits = get_bbox_limits(data, bbox_limits_tolerance)
	bbox_limits = np.asarray(bbox_limits)

	while ((inliers_counter < inliers_num) or (outliers_counter < outliers_num)) and i < points_num:
		noise_list[i] = np.array(random.choice(data))

		# opcion mas rapida np.random.uniform(noise_level[0], noise_level[1], (dim,))
		# aunque no se pueden seleccionar las dimensiones a la que se añaden ruido
		for dim in target_dim:
			noise = np.random.uniform(noise_level[0], noise_level[1])
			noise_list[i][dim] = origin[dim] + noise

		if is_in_bbox(np.asarray([noise_list[i]]), bbox_limits):
			if is_inlier(np.asarray([noise_list[i]]), model_class, model_params, residual_threshold)[0]:
				if inliers_counter < inliers_num:
					inliers_counter += 1
					i += 1
			elif outliers_counter < outliers_num:
				outliers_counter += 1
				i += 1

		trials += 1
		if trials == max_trials:
			raise RuntimeError('Number of trials has been exceeded. Noise has not been generated')
	return np.stack(noise_list, axis = 0)


def gen_gauss_noise(data, model_class, model_params, residual_threshold, n, max_trials, mu, sigma, target_dim, outlier_ratio, seed):
	"""Adds gaussian distributed noise
	Parameters
	----------
	Returns
	-------
	noisy data : (N, dim) array
	"""
	np.random.seed(seed)
	random.seed(seed)

	inliers_num = n * (1 - outlier_ratio)
	outliers_num = n * outlier_ratio

	inliers_counter = 0
	outliers_counter = 0

	target_dim = _check_dim_match(data, target_dim)
	noise_list = [None] * n
	inliers = [True] * n
	residuals = [None] * n
	i = 0
	trials = 0

	while ((inliers_counter < inliers_num) or (outliers_counter < outliers_num)) and i < n:
		noise_list[i] = np.array(random.choice(data))

		for dim in target_dim:
			noise = np.random.normal(mu, sigma)
			noise_list[i][dim] = noise_list[i][dim] + noise

		if is_inlier(np.asarray([noise_list[i]]), model_class, model_params, residual_threshold)[0]:
			if inliers_counter < inliers_num:
				inliers_counter += 1
				i += 1
		elif outliers_counter < outliers_num:
			outliers_counter += 1
			i += 1

		trials += 1
		if trials == max_trials:
			raise RuntimeError('Number of trials has been exceeded. Noise has not been generated')
		#print(f'inliers: {inliers_counter} and outliers: {outliers_counter}')
	return np.stack(noise_list, axis = 0)
	

def gen_outliers(	data, model_class, model_params, residual_threshold, points_num, 
					max_trials, noise_level, cloud_size, target_dim, seed, bbox_limits = None,
					bbox_limits_tolerance = 0):

	"""Adds uniformly distributed noise
	Parameters
	----------
	Returns
	-------
	noisy data : (N, dim) array
	"""
	np.random.seed(seed)
	random.seed(seed)
	target_dim = _check_dim_match(data, target_dim)
	
	first_outlier = gen_gauss_noise(data, model_class, model_params, residual_threshold, 
									points_num = 1, 
									max_trials = max_trials,
									mu = noise_level,
									sigma = noise_level,
									target_dim = target_dim, 
									outlier_ratio = 1.0, 
									seed = seed,
									bbox_limits = bbox_limits,
									bbox_limits_tolerance = bbox_limits_tolerance,
									)

	outliers = gen_uniform_noise(first_outlier, model_class, model_params, residual_threshold, 
								points_num = points_num,
								max_trials = max_trials,
								noise_level = [-cloud_size, cloud_size],
								target_dim = target_dim,
								outlier_ratio = 1.0,
								seed = seed,
								bbox_limits = None,
								bbox_limits_tolerance = bbox_limits_tolerance)

	return outliers

def plane_nd_data(ranges, points_num, seed, origin, normal_vector):
	plane_model = PlaneModelND()
	#ranges = np.array(ranges)*1.5
	data = plane_model.predict(ranges, *points_num, seed, [origin, normal_vector])
	return data
	
def circle_data(n, x_origin, y_origin, radius):
	"""Generate circle data
	Parameters
	----------
	n : int
		Number of points in data.
	x_origin : float
	y_origin : float
	radius : float
	Returns
	-------
	data : array
	"""
	t = np.linspace(0, 2 * np.pi, n)
	data = CircleModel().predict_xy(t, params=(x_origin, y_origin, radius))
	return data

def line_nd_data(axis_range, axis, origin, direction):
	"""Generate circle data
	Parameters
	----------
	n : int
		Number of points in data.
	origin : tuple
	direction : tuple
	Returns
	-------
	data : array
	"""
	x = np.arange(*axis_range, 0.1)
	params = [origin, direction]
	data = LineModelND().predict(x, axis = axis, params = params)
	return data

def ellipse_data(n, x_origin, y_origin, width, height, theta):
	"""Generate circle data
	Parameters
	----------
	n : int
		Number of points in data.
	x_origin : float
	y_origin : float
	height : float
	width : float
	theta: float
		x-axis tilt in degs
	Returns
	-------
	data : array
	"""
	t = np.linspace(0, 2 * np.pi, n)
	data = EllipseModel().predict_xy(t, params=(x_origin, y_origin, width, height, theta))
	return data


if __name__ == '__main__':


	test_num = int(sys.argv[1])
	yaml_file_path = sys.argv[2] + '/yaml/test_'
	
	for test_id in range(1, test_num + 1):

		yaml_file = yaml_file_path + str(test_id) + '.yaml'
		# Read test params
		with open(yaml_file, 'r') as stream:
			test_params = yaml.safe_load(stream)

		# test params
		data_params = test_params['data_params']
		model = data_params['model']
		model_params = data_params['model_params']
		data_len = data_params['data_len']
		data = [[]]

		residual_thresh = test_params['ransac_params'][1] 
		seed = test_params['seed']

		# generate original model data
		if model == 'LineModelND':
			data = line_nd_data(*data_len, *model_params)
			model = LineModelND
		elif model == 'CircleModel':
			data = circle_data(*data_len, *model_params)
			model = CircleModel
		elif model == 'EllipseModel':
			data = ellipse_data(*data_len, *model_params)
			model = EllipseModel

		# noisy data array
		noisy_data = np.array([]).reshape(0, np.shape(data)[1])

		try:
			# add gaussian noise
			gn_params = data_params['gn_params']
			gn_data = [[]]
			# if 'number of noise points required' > 0
			if gn_params[1] > 0:
				gn_data = noise.gen_gauss_noise(data, model, model_params, *gn_params, seed)
				noisy_data = np.vstack([noisy_data, gn_data])

			# add uniform noise
			un_params = data_params['un_params']
			un_data = [[]]
			# if 'number of noise points required' > 0
			if un_params[1] > 0:
				un_data = noise.gen_uniform_noise(data, model, model_params, *un_params, seed)
				noisy_data = np.vstack([noisy_data, un_data])

			# add outlier noise
			on_params = data_params['on_params']
			on_data = [[]]
			# if 'number of noise points required' > 0
			if on_params[1] > 0:
				on_data = noise.gen_outliers(data, model, model_params, *on_params, seed)
				noisy_data = np.vstack([noisy_data, on_data])

			# write noisy data file
			np.savetxt((test_params['save_path'] + '/data/' + test_params['file_name'] + '.txt'), noisy_data)
			# write original data file (for plotting pourposes)
			np.savetxt((test_params['save_path'] + '/data/' + 'original_' + test_params['file_name'] + '.txt'), data)
		
		except RuntimeError:
			print('Data for ' + test_params['file_name'] + ' has not been generated')