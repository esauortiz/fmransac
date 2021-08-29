from test_configuration.utils import _read_yaml
from estimation.results.utils import get_metric

import numpy as np
import sys, os

if __name__ == '__main__':

    model_class = sys.argv[1]
    group_id = sys.argv[2]

    current_path = os.path.dirname(os.path.realpath(__file__))
    scripts_path = current_path[:-19]
    tests_path = _read_yaml(f'{scripts_path}/test_configuration/params/tests_path.yaml')['path']

    # read saved group params (batch_group_params could be modified but tests_path is expected not to change)
    batch_group_params = _read_yaml(f'{tests_path}/{model_class}/00_batch_groups/{group_id}/batch_group_params.yaml')
    initial_batch_id = batch_group_params['group_params']['initial_batch_id']
    n_batches = batch_group_params['group_params']['n_batches']
    estimators_names = batch_group_params['group_params']['estimators_names']
    n_estimators = len(estimators_names)

    # read table params
    table_params = _read_yaml(f'{scripts_path}/estimation/results/table_params.yaml')
    metric = table_params['metric']
    stat_type = table_params['stat_type']

    table = np.empty((n_batches, n_estimators))

    for batch_id in range(n_batches):
        # read batch params
        batch_save_path = f'{tests_path}/{model_class}/batch_{initial_batch_id + batch_id}'
        batch_params = _read_yaml(f'{batch_save_path}/batch_params.yaml')            
        n_tests = batch_params['n_tests']

        batch_row = []
        for estimator in estimators_names:
            #read results
            results = np.loadtxt(f'{batch_save_path}/results/{estimator}/00_{metric}.txt')
            metric_value = get_metric(results, stat_type)
            batch_row.append(metric_value)
        
        table[batch_id] = batch_row
    
    # save table
    np.savetxt(f'{tests_path}/{model_class}/00_batch_groups/{group_id}/results_tables/{metric}_{stat_type}.txt', table)