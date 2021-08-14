from test_configuration.utils import _read_yaml, _create_directory
import os, io, yaml

if __name__ == "__main__":

    print('[*] Saving batch group configuration ...')

    current_path = os.path.dirname(os.path.realpath(__file__))
    batch_group_params = _read_yaml(f'{current_path}/params/batch_group.yaml')
    tests_path = _read_yaml(f'{current_path}/params/tests_path.yaml')['path']
    model_class = batch_group_params['group_params']['model_class']
    group_id = batch_group_params['group_params']['group_id']

    _create_directory(f'{tests_path}/{model_class}')
    _create_directory(f'{tests_path}/{model_class}/00_batch_groups')
    _create_directory(f'{tests_path}/{model_class}/00_batch_groups/{group_id}')
    _create_directory(f'{tests_path}/{model_class}/00_batch_groups/{group_id}/results_tables')

    with io.open(f'{tests_path}/{model_class}/00_batch_groups/{group_id}/batch_group_params.yaml', 'w', encoding='utf8') as outfile:
        yaml.dump(batch_group_params, outfile, default_flow_style=False, allow_unicode=True)

    print(f'    Batch group `{group_id}` parameters saved in {tests_path}/{model_class}/00_batch_groups/{group_id}/batch_group_params.yaml')
