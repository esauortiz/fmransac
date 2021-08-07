import yaml
import io
import os

from scipy.stats.distributions import chi2
from file_utils import _read_yaml

if __name__ == '__main__':

	# batch group params
	current_path = os.path.dirname(os.path.realpath(__file__))
	batch_group_params = _read_yaml(f'{current_path}/params/batch_group.yaml')
	save_path 		 = batch_group_params['group_params']['save_path']
	initial_batch_id = batch_group_params['group_params']['initial_batch_id']
	model_class 	 = batch_group_params['group_params']['model_class']
	n_batches 		 = batch_group_params['group_params']['n_batches']
	n_tests 		 = batch_group_params['group_params']['n_tests']

	# model class params
	model_params = _read_yaml(f'{current_path}/params/model/{model_class}.yaml')

	# dataset params
	dataset_params = batch_group_params['dataset_params']
		# all params in dataset_params as list
	for key in dataset_params:
		if not isinstance(dataset_params[key], list) and key != 'noise_dim':
			param_list = []
			for i in range(n_batches):
				param_list.append(dataset_params[key])
			dataset_params[key] = param_list
	
	# ransac params
	ransac_params = batch_group_params['ransac_params']
		# all params in dataset_params as list
	for key in ransac_params:
		if not isinstance(ransac_params[key], list) and key != 'noise_dim':
			param_list = []
			for i in range(n_batches):
				param_list.append(ransac_params[key])
			ransac_params[key] = param_list

		# if kappa != 0 then thereshold values will be computed 
		# based on dataset standard deviation (sd)
	if ransac_params['kappa'][0] != 0:
		threshold = []
		print('Thereshold values will be computed based on dataset standard deviation (sd)')
		for kappa, sd in zip(ransac_params['kappa'], dataset_params['sd']):
			threshold.append(kappa*sd)
		ransac_params['threshold'] = threshold

		# if inlier_prob != 0 then thereshold values will be computed 
		# assuming a chi2 distribution of the fitting errors
	if ransac_params['inlier_prob'][0] != 0:
		threshold = []
		print('Thereshold values will be computed assuming a chi2 distribution of the fitting errors')
		for inlier_prob, df, sd in zip(ransac_params['inlier_prob'], ransac_params['df'], dataset_params['sd']):
			threshold.append(((sd**2)*float(chi2.ppf(0.997, df=sd)))**0.5)
		ransac_params['threshold'] = threshold

		# if theta == 0 then threshold values are chosen as theta values
	if ransac_params['theta'][0] == 0:
		print('Threshold values are chosen as theta values')
		ransac_params['theta'] = ransac_params['threshold']

	for i in range(n_batches):
		batch_id = f'batch_{initial_batch_id + i}'
		batch_params = {
			'save_path' : f'{save_path}/{model_class}/{batch_id}',
			'batch_id' : batch_id,
			'n_tests' : n_tests,
			'model_params' : model_params,
			'dataset_params' : {
				'model_bbox' 	: dataset_params['model_bbox'][i],
				'dataset_bbox' 	: dataset_params['dataset_bbox'][i],
				'outlier_ratio' : dataset_params['outlier_ratio'][i],
				'n_points' 		: dataset_params['n_points'][i],
				'noise_dim' 	: dataset_params['noise_dim'],
				'sd' 			: dataset_params['sd'][i],
				'uniform_noise_bbox' : dataset_params['uniform_noise_bbox'][i],
			},
			'ransac_params' : {
				'outlier_ratio' : dataset_params['outlier_ratio'][i],
				't_max' 		: ransac_params['t_max'][i],
				'stop_prob' 	: ransac_params['stop_prob'][i],
				'min_samples' 	: ransac_params['min_samples'][i],
				'threshold' 	: ransac_params['threshold'][i],
				'theta' 		: ransac_params['theta'][i],
				'n' 			: ransac_params['n'][i]
			}
		}

		# Write YAML file
		with io.open(f'{save_path}/{model_class}/{batch_id}/batch_params.yaml', 'w', encoding='utf8') as outfile:
			yaml.dump(batch_params, outfile, default_flow_style=False, allow_unicode=True)