"""
# Usage
    python scripts/estimation/results/tabulate.py model_class group_id rows_labels metric stat_type

# Args
    model_class: model class could take values in ['PlaneModelND', 'EllipseModel', 'HomographyModel']
    group_id: group identifier beginning with G
    rows_labels: indicates which batch parameters should be used as rows labels in data frame. Must match number of batches (n_batches)
    metric: results metric could take values in ['abs_errors', 'rel_errors', 'estimation_errors', 'iterations']
    stat_type: statistic value based on results metric. Takes any percentile beginning with P (e.g. 'P95'), 'mean' or 'sd'

# Example
    python scripts/estimation/results/tabulate.py PlaneModelND G1 outlier_ratio estimation_errors P95
"""

from test_configuration.utils import _read_yaml
from estimation.results.utils import get_metric
from estimation.utils import get_residuals
from estimation.fit import HomographyModel
from pathlib import Path
import numpy as np
import pandas as pd
import sys

def main():
    model_class = sys.argv[1]
    group_id = sys.argv[2]
    param_as_row_label = sys.argv[3]
    metric = sys.argv[4]
    stat_type = sys.argv[5]
    
    current_path = Path(__file__).parent.resolve()
    scripts_path = current_path.parent.parent
    tests_path = _read_yaml(f'{scripts_path}/test_configuration/params/tests_path.yaml')['path']

    # read saved group params (batch_group_params could be modified but tests_path is expected not to change)
    batch_group_params = _read_yaml(f'{tests_path}/{model_class}/00_batch_groups/{group_id}/batch_group_params.yaml')
    initial_batch_id = batch_group_params['group_params']['initial_batch_id']
    n_batches = batch_group_params['group_params']['n_batches']
    estimators_names = batch_group_params['group_params']['estimators_names']
    n_estimators = len(estimators_names)
    residual_threshold = batch_group_params['ransac_params']['residual_threshold']

    # build results as data frame
    data = np.empty((n_batches, n_estimators + 1))
    row_labels = []

    for batch_id in reversed(range(n_batches)):
        # read batch params
        batch_save_path = f'{tests_path}/{model_class}/batch_{initial_batch_id + batch_id}'
        batch_params = _read_yaml(f'{batch_save_path}/batch_params.yaml')            
        try:
            row_labels.append(batch_params['dataset_params'][param_as_row_label])
        except KeyError:
            row_labels.append(batch_params['ransac_params'][param_as_row_label])
            
        batch_row = []
        for estimator in estimators_names:
            #read results
            results = np.loadtxt(f'{batch_save_path}/results/{estimator}/00_{metric}.txt')
            metric_value = get_metric(results, stat_type)
            batch_row.append(metric_value)
        
        # number of inliers in data generated with images + features detector + features matching
        params_original = np.loadtxt(f'{batch_save_path}/tests_params/original_params.txt')
        whole_data2 = np.loadtxt(f'{batch_save_path}/datasets/dst_pts.txt')
        whole_data1 = np.loadtxt(f'{batch_save_path}/datasets/src_pts.txt')
        whole_data = np.column_stack((whole_data1,whole_data2))
        residuals = get_residuals(whole_data, HomographyModel, params_original)
        whole_original_inliers = residuals < residual_threshold
        n_true_inliers = np.append(n_true_inliers, np.sum(whole_original_inliers))
        batch_row.append(n_true_inliers)

        data[-(batch_id + 1)] = batch_row # -(batch_id + 1) because reversed

    results_df = pd.DataFrame(data, index=row_labels, columns=[*estimators_names, 'n_true_inliers'])
    
    # save table
    print(results_df)
    results_df.to_csv(f'{tests_path}/{model_class}/00_batch_groups/{group_id}/results_tables/{metric}_{stat_type}.csv')
    print(f'Data frame has been saved in {tests_path}/{model_class}/00_batch_groups/{group_id}/results_tables/{metric}_{stat_type}.csv')

if __name__ == '__main__':
    main()