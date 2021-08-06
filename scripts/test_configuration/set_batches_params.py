import yaml
import io
import os

from scipy.stats.distributions import chi2

def _read_yaml(file_name):
	with open(file_name, 'r') as stream: 
		return yaml.safe_load(stream)

if __name__ == '__main__':

	# batch group params
	current_path = os.path.dirname(os.path.realpath(__file__))
	batch_group = _read_yaml(f'{current_path}/params/batch_group.yaml')
	save_path 		 = batch_group['group_params']['save_path']
	initial_batch_id = batch_group['group_params']['initial_batch_id']
	model_class 	 = batch_group['group_params']['model_class']
	n_batches 		 = batch_group['group_params']['n_batches']

	# model class params
	model_params = _read_yaml(f'{current_path}/params/model/{model_class}.yaml')

	# dataset params
	dataset_params = batch_group['dataset_params']
		# all params in dataset_params as list
	for key in dataset_params:
		if not isinstance(dataset_params[key], list) and key != 'noise_dim':
			param_list = []
			for i in range(n_batches):
				param_list.append(dataset_params[key])
			dataset_params[key] = param_list

	inliers_num = [int(dataset_params['n_points'][0]*int(1-x)) for x in dataset_params['outlier_ratio']]
	outliers_num = [int(dataset_params['n_points'][0]*(x)) for x in dataset_params['outlier_ratio']]
	uniform_noise_bbox = dataset_params['dataset_bbox']

	# ransac params
	ransac_params = batch_group['ransac_params']
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
			'model_params' : model_params,
			'dataset_params' : {
				'model_bbox' 	: dataset_params['model_bbox'][i],
				'dataset_bbox' 	: dataset_params['dataset_bbox'][i],
				'noise_dim' 	: dataset_params['noise_dim'],
				'outlier_ratio' : dataset_params['outlier_ratio'][i],
				'sd' 			: dataset_params['sd'][i],
				'n_points' 		: dataset_params['n_points'][i],
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