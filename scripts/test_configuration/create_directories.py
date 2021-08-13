from test_configuration.utils import _read_yaml, _create_directory, _remove_directory
import numpy as np
import os, io, yaml

if __name__ == "__main__":

	print('[*] Creating directories ...')

    # read batch configuration
	current_path = os.path.dirname(os.path.realpath(__file__))
	yaml_file = f'{current_path}/params/batch_group.yaml'
	batch_group_params = _read_yaml(yaml_file)
	save_path = batch_group_params['group_params']['save_path']
	n_batches = batch_group_params['group_params']['n_batches']
	initial_batch_id = batch_group_params['group_params']['initial_batch_id']
	model_class = batch_group_params['group_params']['model_class']
	estimators = batch_group_params['group_params']['estimators_names']
	group_id = batch_group_params['group_params']['group_id']

	remove_directories = False
	if input("Do you want to remove previously created directories? (y/n): ") == 'y':
		remove_directories = True

	created_directories = []

	_create_directory(f'{save_path}/{model_class}', created_directories)
	_create_directory(f'{save_path}/{model_class}/00_batch_groups', created_directories)
	_create_directory(f'{save_path}/{model_class}/00_batch_groups/{group_id}', created_directories)
	_create_directory(f'{save_path}/{model_class}/00_batch_groups/{group_id}/results_tables', created_directories)
	
	with io.open(f'{save_path}/{model_class}/00_batch_groups/{group_id}/batch_group_params.yaml', 'w', encoding='utf8') as outfile:
		yaml.dump(batch_group_params, outfile, default_flow_style=False, allow_unicode=True)
	print(f'Batch group `{group_id}` parameters saved in {save_path}/{model_class}/00_batch_groups/{group_id}/batch_group_params.yaml')

	for batch_id in range(n_batches):
		batch_save_path = f'{save_path}/{model_class}/batch_{initial_batch_id + batch_id}'
		print(f'Directory: {batch_save_path} created')

		if remove_directories == True:
			_remove_directory(batch_save_path)
		
		_create_directory(batch_save_path, created_directories)
		_create_directory(f'{batch_save_path}/datasets', created_directories)
		_create_directory(f'{batch_save_path}/tests_params', created_directories)
		_create_directory(f'{batch_save_path}/results', created_directories)
		for estimator in estimators:
			_create_directory(f'{batch_save_path}/results/{estimator}', created_directories)
	"""
	with open(f'{save_path}/{model_class}/00_batch_groups/{group_id}/created_directories.txt', 'w') as f:
		for dir in created_directories:
			f.write("%s\n" % dir)
	"""