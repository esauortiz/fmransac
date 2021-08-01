import numpy as np
import random
import sys
import yaml
import io
from multiprocessing import Pool

from estimation.myfit import PlaneModelND, is_inlier, get_residuals
from gen_data import plane_nd_data, get_bbox_limits, is_in_bbox, gen_uniform_noise

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

def gen_gauss_noise(data, model_class, model_params, residual_threshold, points_num, 
					max_trials, mu, sigma, target_dim, outlier_ratio, seed, bbox_limits = None, 
					bbox_limits_tolerance = 0):
	"""Adds gaussian distributed noise
	Parameters
	----------
	Returns
	-------
	noisy data : (N, dim) array
	"""
	np.random.seed(seed)
	random.seed(seed)
	points_num = int(points_num)

	origin, normal_vector = model_params
	origin = np.asarray(origin)
	normal_vector = np.asarray(normal_vector)
	normal_vector /= np.linalg.norm(normal_vector)
	dim = np.size(normal_vector)

	inliers_num = points_num * (1 - outlier_ratio)
	outliers_num = points_num * outlier_ratio

	inliers_counter = 0
	outliers_counter = 0
	points_num = int(points_num)

	target_dim = _check_dim_match(data, target_dim)
	noise_list = np.empty((points_num, dim), dtype = float)
	i = 0
	trials = 0

	if bbox_limits is None:
		bbox_limits = get_bbox_limits(data, bbox_limits_tolerance)
	bbox_limits = np.asarray(bbox_limits)

	while ((inliers_counter < inliers_num) or (outliers_counter < outliers_num)) and i < points_num:
		
		noise_list[i] = np.array(random.choice(data))
		noise = np.random.normal(mu, sigma)
		noise_list[i] += (normal_vector * noise)

		if is_in_bbox(np.asarray([noise_list[i]]), bbox_limits):
			if is_inlier(np.asarray([noise_list[i]]), model_class, model_params, residual_threshold):
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
	

if __name__ == '__main__':


	test_num = int(sys.argv[1])
	yaml_file_path = sys.argv[2] + '/yaml/test_'
	
	def generate_data(test_id):

		yaml_file = yaml_file_path + str(test_id) + '.yaml'
		# Read test params
		with open(yaml_file, 'r') as stream:
			test_params = yaml.safe_load(stream)

		# test params
		data_params = test_params['data_params']
		model_params = data_params['model_params']
		data_len = data_params['data_len']
		data = [[]]

		#seed = test_params['seed']
		seed = test_params['seed']

		# generate original model data
		# test_id as random seed
		data = plane_nd_data(*data_len, test_id, *model_params)
		model_class = PlaneModelND

		# noisy data array
		noisy_data = np.array([]).reshape(0, np.shape(data)[1])
		try:
			# add gaussian noise
			gn_params = data_params['gn_params']
			gn_data = [[]]
			# if 'number of noise points required' > 0
			if gn_params[1] > 0:
				gn_data = gen_gauss_noise(data, model_class, model_params, *gn_params, seed, bbox_limits = data_params['bbox_limits'], bbox_limits_tolerance = data_params['bbox_limits_tolerance'])
				noisy_data = np.vstack([noisy_data, gn_data])
		
			# add uniform noise
			un_params = data_params['un_params']
			un_data = [[]]
			# if 'number of noise points required' > 0
			if un_params[1] > 0:
				un_data = gen_uniform_noise(data, model_class, model_params, *un_params, seed, bbox_limits = data_params['bbox_limits'], bbox_limits_tolerance = data_params['bbox_limits_tolerance'])
				noisy_data = np.vstack([noisy_data, un_data])

			# add outlier noise
			on_params = data_params['on_params']
			on_data = [[]]
			# if 'number of noise points required' > 0
			if on_params[1] > 0:
				on_data = gen_outliers(data, model_class, model_params, *on_params, seed, bbox_limits = data_params['bbox_limits'], bbox_limits_tolerance = data_params['bbox_limits_tolerance'])
				noisy_data = np.vstack([noisy_data, on_data])

			# write noisy data file
			np.savetxt((test_params['save_path'] + '/data/' + test_params['file_name'] + '.txt'), noisy_data)
			# write original data inliers
			residuals = get_residuals(noisy_data, model_class, model_params)
			inliers = residuals < gn_params[0] # all inlier data
			np.savetxt((test_params['save_path'] + '/data/' + test_params['file_name'] + '_inliers.txt'), inliers)
			np.savetxt((test_params['save_path'] + '/data/' + test_params['file_name'] + '_residuals.txt'), residuals)
			# write original data file (for plotting pourposes)
			#np.savetxt((test_params['save_path'] + '/data/' + 'original_' + test_params['file_name'] + '.txt'), data)
		
		except RuntimeError:
			print('Data for ' + test_params['file_name'] + ' has not been generated')

		return True

	#generate_data(test_num)
	with Pool(8) as p:
		p.map(generate_data, range(1, test_num + 1))
