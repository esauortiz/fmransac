from test_configuration.utils import _read_yaml, _create_directory, _remove_directory
import numpy as np
import sys, os, io, yaml

if __name__ == "__main__":

	print('[*] Creating directories ...')

	model_class = sys.argv[1]
	group_id = sys.argv[2]

    # read batch configuration
	current_path = os.path.dirname(os.path.realpath(__file__))
	tests_path = _read_yaml(f'{current_path}/params/tests_path.yaml')['path']
	batch_group_params = _read_yaml(f'{tests_path}/{model_class}/00_batch_groups/{group_id}/batch_group_params.yaml')
	n_batches = batch_group_params['group_params']['n_batches']
	initial_batch_id = batch_group_params['group_params']['initial_batch_id']
	estimators = batch_group_params['group_params']['estimators_names']

	remove_directories = False
	if input("    Do you want to remove previously created batch directories? (y/n): ") == 'y':
		remove_directories = True

	created_directories = []

	for batch_id in range(n_batches):
		batch_save_path = f'{tests_path}/{model_class}/batch_{initial_batch_id + batch_id}'
		print(f'    Directory created: {batch_save_path}')

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