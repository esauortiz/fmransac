import os
from test_configuration.utils import _read_yaml, _create_directory, _remove_directory

if __name__ == "__main__":

    # read batch configuration
	current_path = os.path.dirname(os.path.realpath(__file__))
	yaml_file = f'{current_path}/params/batch_group.yaml'
	batch_group_params = _read_yaml(yaml_file)
	save_path = batch_group_params['group_params']['save_path']
	n_batches = batch_group_params['group_params']['n_batches']
	initial_batch_id = batch_group_params['group_params']['initial_batch_id']
	model_class = batch_group_params['group_params']['model_class']
	estimators = batch_group_params['group_params']['estimators_names']

	remove_directories = False
	if input("Do you want to remove previously created directories? (y/n): ") == 'y':
		remove_directories = True

	_create_directory(f'{save_path}/{model_class}')

	for batch_id in range(n_batches):
		batch_save_path = f'{save_path}/{model_class}/batch_{initial_batch_id + batch_id}'

		if remove_directories == True:
			_remove_directory(batch_save_path)

		_create_directory(batch_save_path)
		_create_directory(f'{batch_save_path}/datasets')
		_create_directory(f'{batch_save_path}/tests_params')
		_create_directory(f'{batch_save_path}/results')
		for estimator in estimators:
			_create_directory(f'{batch_save_path}/results/{estimator}')