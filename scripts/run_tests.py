from test_configuration.utils import _read_yaml
import sys, os

if __name__ == '__main__':

    model_class = sys.argv[1]
    group_id = sys.argv[2]
    current_path = os.path.dirname(os.path.realpath(__file__))
    tests_path = _read_yaml(f'{current_path}/test_configuration/params/tests_path.yaml')['path']
    # read saved group params (batch_group_params could be modified but save_path is expected not to change)
    batch_group_params = _read_yaml(f'{tests_path}/{model_class}/00_batch_groups/{group_id}/batch_group_params.yaml')
    initial_batch_id = batch_group_params['group_params']['initial_batch_id']
    n_batches = batch_group_params['group_params']['n_batches']

    # tests configuration
    os.system(f'python  {current_path}/test_configuration/create_directories.py {model_class} {group_id}')
    os.system(f'python  {current_path}/test_configuration/set_batches_params.py {model_class} {group_id}')
    os.system(f'python  {current_path}/test_configuration/set_tests_params.py {model_class} {group_id}')

    # dataset generation
    print('[*] Generating datasets ...')
    for batch_id in range(n_batches):
        os.system(f'python  {current_path}/dataset_generation/main.py {model_class} batch_{initial_batch_id + batch_id}')

    # estimation
    print('[*] Estimating parameters ...')
    for batch_id in range(n_batches):
        os.system(f'python  {current_path}/estimation/main.py {model_class} batch_{initial_batch_id + batch_id}')

    print(f'[*] Generating results ...')
    for batch_id in range(n_batches):
        os.system(f'python  {current_path}/estimation/results.py {model_class} batch_{initial_batch_id + batch_id}')
