from dataset_generation.utils import _is_inlier, _is_in_bbox, _check_dim_match, _get_bbox_from_value
from estimation.fit import PlaneModelND
import numpy as np
import random

def _gaussian_noise(data, original_model, dataset_params, seed = None):
	""" Add noise points whose fitting errors 
	follows a Gaussian distribution
	Parameters
	----------
	data : (N, dim) array
	original_model : model_class
	dataset_params : list
	seed : int
	Returns
	-------
	noisy_data : (N, dim) array
	"""
	np.random.seed(seed)
	random.seed(seed)

	origin, normal_vector = original_model.params
	normal_vector /= np.linalg.norm(normal_vector)
	dataset_dim = np.size(normal_vector)
	dataset_bbox = _get_bbox_from_value(dataset_params['dataset_bbox'], dataset_dim)

	n_points = int(dataset_params['n_points'])
	n_inliers = int(n_points * (1 - dataset_params['outlier_ratio']))
	residual_threshold = dataset_params['sd'] * dataset_params['residual_kappa']

	_check_dim_match(data, dataset_params['noise_dim'])
	noisy_data = np.empty((n_inliers, dataset_dim), dtype = float)

	i = 0
	trials = 0

	while i < n_inliers:
		
		noisy_data[i] = np.array(random.choice(data))
		gaussian_sample = np.random.normal(0, dataset_params['sd'])
		noisy_data[i] += (normal_vector * gaussian_sample)

		if _is_in_bbox(noisy_data[i], dataset_bbox):
			if _is_inlier(noisy_data[i], original_model, residual_threshold):
				i += 1
		
		trials += 1

		if trials == dataset_params['max_trials']:
			raise RuntimeError('Number of trials has been exceeded. Gaussian noise has not been generated')
			
	#return np.stack(noisy_data, axis = 0)
	return noisy_data

def _get_model_dataset(model_params, model_samples, model_bbox, seed = None, print_model_params = False, output = 'screen'):
	model = PlaneModelND(model_params)
	dim = len(model_params[0]) # dim = number of components of origin
	model_bbox = _get_bbox_from_value(model_bbox, dim)
	if print_model_params == True:
		Ao, Bo, Co = model.get_general_params(model_params)
		if output == 'screen' : print(f'Model params: {"%.2f" % Ao}x + {"%.2f" % Bo}y + {"%.2f" % Co} = 0')
	dataset = model.predict(model_bbox, model_samples, seed = seed)
	return dataset[dataset[:, 1].argsort(kind='mergesort')]
if __name__ == '__main__':
	True