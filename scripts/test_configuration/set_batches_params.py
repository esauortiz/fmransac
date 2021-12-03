from scipy.stats.distributions import chi2
from test_configuration.utils import _read_yaml
import io, os, sys, yaml

if __name__ == '__main__':

	print('[*] Setting batch parameters ...')
	
	model_class = sys.argv[1]
	group_id = sys.argv[2]

	# batch group params
	current_path = os.path.dirname(os.path.realpath(__file__))
	tests_path = _read_yaml(f'{current_path}/params/tests_path.yaml')['path']
	batch_group_params = _read_yaml(f'{tests_path}/{model_class}/00_batch_groups/{group_id}/batch_group_params.yaml')
	initial_batch_id = batch_group_params['group_params']['initial_batch_id']
	model_class 	 = batch_group_params['group_params']['model_class']
	n_batches 		 = batch_group_params['group_params']['n_batches']
	n_tests 		 = batch_group_params['group_params']['n_tests']

	# model class params
	model_params = _read_yaml(f'{current_path}/params/model/{model_class}.yaml')

	# dataset params
	dataset_params = batch_group_params['dataset_params']
		# all params in dataset_params as list, repeating values if a single values instead of a list is provided
		# i.e. if min_samples = 3 it will be converted to min_samples = np.repeat(3, n_batches).tolist()
	if not dataset_params['real_dataset']:
		for key in dataset_params:
			if not isinstance(dataset_params[key], list) and key not in ['noise_dim', 'real_dataset']:
				param_list = []
				for i in range(n_batches):
					param_list.append(dataset_params[key])
				dataset_params[key] = param_list
	else:
		# instead of synthetic dataset params, place label with batch_id identifier
		# i.e. each batch will have dataset_label = '{dataset_label}_{batch_id}'
		# in order to place in each batch a different unique dataset, changing only
		# the sequence of mss selected in the main loop for each estimation test   
		labels_list = []
		for i in range(n_batches):
			labels_list.append(dataset_params['dataset_label'] + '_' + str(i))
		dataset_params['dataset_label']	= labels_list
		
		# add outlier_ratio to set ransac main loop iterations
		if not isinstance(dataset_params['outlier_ratio'], list):
			or_list = []
			for i in range(n_batches):
				or_list.append(dataset_params['outlier_ratio'])
			dataset_params['outlier_ratio'] = or_list



	# ransac params
	ransac_params = batch_group_params['ransac_params']
		# all params in dataset_params as list
	for key in ransac_params:
		if not isinstance(ransac_params[key], list):
			param_list = []
			for i in range(n_batches):
				param_list.append(ransac_params[key])
			ransac_params[key] = param_list

		# if kappa != 0 then thereshold values will be computed 
		# based on dataset standard deviation (sd)
	if ransac_params['residual_kappa'][0] != 0 and not dataset_params['real_dataset']:
		threshold = []
		print('    Thereshold values will be computed based on dataset `standard deviation` and `kappa` multiplier')
		for kappa, sd in zip(ransac_params['residual_kappa'], dataset_params['sd']):
			threshold.append(kappa*sd)
		ransac_params['residual_threshold'] = threshold

		# if inlier_prob != 0 then thereshold values will be computed 
		# assuming a chi2 distribution of the fitting errors
	if ransac_params['inlier_prob'][0] != 0 and not dataset_params['real_dataset']:
		threshold = []
		print('    Thereshold values will be computed assuming a chi2 distribution of the fitting errors')
		for inlier_prob, df, sd in zip(ransac_params['inlier_prob'], ransac_params['df'], dataset_params['sd']):
			threshold.append(((sd**2)*float(chi2.ppf(0.997, df=sd)))**0.5)
		ransac_params['residual_threshold'] = threshold

		# if theta == 0 then threshold values are chosen as theta values
	if ransac_params['theta'][0] == 0:
		print('    Threshold values are chosen as theta values')
		ransac_params['theta'] = ransac_params['residual_threshold']

	for i in range(n_batches):
		batch_id = f'batch_{initial_batch_id + i}'
		batch_params = {
			'batch_id' : batch_id,
			'n_tests' : n_tests,
			'estimators_names' : batch_group_params['group_params']['estimators_names'],
			'ransac_params' : {
				'outlier_ratio' : dataset_params['outlier_ratio'][i],
				'max_trials' 	: ransac_params['max_trials'][i],
				't_max' 		: ransac_params['t_max'][i],
				'stop_prob' 	: ransac_params['stop_prob'][i],
				'min_samples' 	: ransac_params['min_samples'][i],
				'theta' 		: ransac_params['theta'][i],
				'sigma_phi' 	: ransac_params['sigma_phi'][i],
				'n' 			: ransac_params['n'][i],
				'residual_threshold' : ransac_params['residual_threshold'][i],
				'convergence_threshold' : ransac_params['convergence_threshold'][i]
			}
		}
		# if synthetic data is generated place in 'dataset_params' key its description
		if not dataset_params['real_dataset']:
			batch_params['dataset_params'] = {
				'model_bbox' 	: dataset_params['model_bbox'][i],
				'dataset_bbox' 	: dataset_params['dataset_bbox'][i],
				'outlier_ratio' : dataset_params['outlier_ratio'][i],
				'n_points' 		: dataset_params['n_points'][i],
				'noise_dim' 	: dataset_params['noise_dim'],
				'sd' 			: dataset_params['sd'][i],
				'uniform_noise_bbox' : dataset_params['uniform_noise_bbox'][i],
				'residual_kappa' : dataset_params['residual_kappa'][i],
				'max_trials' : dataset_params['max_trials'][i],
			}
			batch_params['model_params'] = model_params
		# if data is provided (e.g. real data) only place a label for the unique dataset placed in each batch
		else:
			batch_params['dataset_params'] = {
				'label': dataset_params['dataset_label'][i],
				'real_dataset' : dataset_params['real_dataset']
				}
			batch_params['model_params'] = {'model_class': model_class}

		# Write YAML file
		with io.open(f'{tests_path}/{model_class}/{batch_id}/batch_params.yaml', 'w', encoding='utf8') as outfile:
			yaml.dump(batch_params, outfile, default_flow_style=False, allow_unicode=True)