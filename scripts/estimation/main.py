from estimation.estimators import RANSAC, MSAC, FMR
from estimation.fuzzy_metrics import M1, M2, M3, M4
from estimation.fit import PlaneModelND, EllipseModel, HomographyModel
from test_configuration.utils import _read_yaml
from multiprocessing import Pool
import numpy as np
import sys, os

if __name__ == '__main__':

    model_class = str(sys.argv[1])
    batch_id = str(sys.argv[2])

    current_path = os.path.dirname(os.path.realpath(__file__))
    scripts_path = current_path[:-11]
    yaml_file = f'{scripts_path}/test_configuration/params/batch_group.yaml'
    batch_group_params = _read_yaml(yaml_file)
    save_path = batch_group_params['group_params']['save_path']

    batch_params = _read_yaml(f'{save_path}/{model_class}/{batch_id}/batch_params.yaml')
    dataset_params = batch_params['dataset_params']
    save_path = batch_params['save_path']
    n_tests = batch_params['n_tests']
    model_class = eval(batch_params['model_params']['model_class'])
    estimators_names = batch_params['estimators_names']
    ransac_params = batch_params['ransac_params']
    outlier_ratio = batch_params['dataset_params']['outlier_ratio']

    # configure_ estimators
    min_samples = ransac_params['min_samples']
    residual_threshold = ransac_params['residual_threshold']
    max_trials = ransac_params['max_trials']
    t_max = ransac_params['t_max']
    stop_probability = ransac_params['stop_prob']
    convergence_threshold = ransac_params['convergence_threshold']
    theta = ransac_params['theta']
    n = ransac_params['n']
    sigma_phi = ransac_params['sigma_phi']

    estimators = []
    for estimator_name in estimators_names:
        if estimator_name not in ['default', 'RANSAC', 'MSAC']:
            variant = int(estimator_name[3])
            fuzzy_metric = eval(estimator_name[-2:])(n = n, theta = theta)
            estimators.append(FMR( min_samples, residual_threshold, max_trials, stop_probability, 
                                    variant, fuzzy_metric, t_max, sigma_phi, convergence_threshold, outlier_ratio))
        elif estimator_name == 'RANSAC':
            estimators.append(RANSAC(  min_samples, residual_threshold, max_trials, 
                                        stop_probability, outlier_ratio))
        elif estimator_name == 'MSAC':
            estimators.append(MSAC(min_samples, residual_threshold, max_trials, 
                                    stop_probability, outlier_ratio))

    def _run_test(test_id):
        sys.stdout.write("\rEstimating params for test %i" % test_id)
        sys.stdout.flush()
        test_id = str(test_id)

        # Read noisy data
        if model_class.__name__ == 'HomographyModel':
            try:
                data1 = np.loadtxt(f'{save_path}/datasets/test_{test_id}_proj1.txt')
                data2 = np.loadtxt(f'{save_path}/test_{test_id}_proj2.txt')
                data = np.column_stack((data1,data2))
            except IOError:
                print(f'Data for test_{test_id} not found')
                #continue
        else:
            try:
                data = np.loadtxt(f'{save_path}/datasets/test_{test_id}.txt', delimiter=" ")
                #data = data[~np.isnan(data)]
            except IOError:
                print(f'Data for test_{test_id} not found')
                return True
            
        # default estimation (e.g. Total Least Squares)
        # fit line using all data
        model_d = model_class()
        model_d.estimate(data)
        #np.savetxt((save_path + 'default/test_' + test_id + '_params.txt'), model_d.params)

        for estimator_name, estimator in zip(estimators_names, estimators):
            # robustly fit line only using inlier data (robust model)
            model, inliers, scores, iterations = estimator.run(data, model_class, seed = test_id)
            if model is not None:
                np.savetxt(f'{save_path}/results/{estimator_name}/test_{test_id}_inliers.txt', inliers)
                np.savetxt(f'{save_path}/results/{estimator_name}/test_{test_id}_params.txt', model.params)
                if estimator_name not in ['RANSAC', 'MSAC']:
                    np.savetxt(f'{save_path}/results/{estimator_name}/test_{test_id}_iterations.txt', np.array([iterations]))
                    np.savetxt(f'{save_path}/results/{estimator_name}/test_{test_id}_scores.txt', scores)
        return True

    #for test_id in range(n_tests):
    #    _run_test(test_id)
    with Pool(8) as p:
        p.map(_run_test, range(n_tests))