from estimation.estimators import RANSAC, MSAC, FMR
from estimation.fuzzy_metrics import M1, M2, M3, M4
from estimation.fit import PlaneModelND, EllipseModel, HomographyModel
from estimation.utils import _check_data_atleast_2D
from test_configuration.utils import _read_yaml
from multiprocessing import Pool, Value
import numpy as np
import sys, os

if __name__ == '__main__':

    model_class = str(sys.argv[1])
    batch_id = str(sys.argv[2])

    # read batch params
    current_path = os.path.dirname(os.path.realpath(__file__))
    scripts_path = current_path[:-11]
    tests_path = _read_yaml(f'{scripts_path}/test_configuration/params/tests_path.yaml')['path']

    batch_save_path = f'{tests_path}/{model_class}/{batch_id}'
    batch_params = _read_yaml(f'{batch_save_path}/batch_params.yaml')
    dataset_params = batch_params['dataset_params']
    n_tests = batch_params['n_tests']
    model_class = eval(batch_params['model_params']['model_class'])
    estimators_names = batch_params['estimators_names']
    ransac_params = batch_params['ransac_params']
    outlier_ratio = batch_params['dataset_params']['outlier_ratio']

    # configure estimators
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
        if estimator_name not in ['RANSAC', 'MSAC', 'default']:
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
        elif estimator_name == 'default':
            estimators.append(None)

    def _run_test(test_id):
        global finished_tests

        # Read noisy data
        if model_class.__name__ == 'HomographyModel':
            try:
                data1 = np.loadtxt(f'{batch_save_path}/datasets/test_{test_id}_proj1.txt')
                data2 = np.loadtxt(f'{batch_save_path}/datasets/test_{test_id}_proj2.txt')
                data = np.column_stack((data1,data2))
            except IOError:
                print(f'    Data for test_{test_id} not found')
                #continue
        else:
            try:
                raw_data = np.loadtxt(f'{batch_save_path}/datasets/test_{test_id}.txt', delimiter=" ")
                # if a point contains NaN values won't be included in data
                data = []
                for point in raw_data:
                    if np.sum(np.isnan(point)) > 0:
                        continue
                    else:
                        data.append(point)
                data = np.array(data)
                # filter selecting less data
                samples = np.linspace(0, data.shape[0], 500, dtype=int)
                data = data[samples]
            except IOError:
                print(f'    Data for test_{test_id} not found')
                return True
            
        try:
            _check_data_atleast_2D(data)
        except ValueError:
            return True

        for estimator_name, estimator in zip(estimators_names, estimators):
            # default estimation (e.g. Total Least Squares)
            # fit line using all data
            if estimator_name == 'default':
                model = model_class()
                model.estimate(data)
                model_residuals = model.residuals(data)
                np.savetxt(f'{batch_save_path}/results/{estimator_name}/test_{test_id}_params.txt', model.params)
                np.savetxt(f'{batch_save_path}/results/{estimator_name}/test_{test_id}_residuals.txt', model_residuals)
            else:
                # robustly fit line only using inlier data (robust model)
                model, inliers, scores, iterations = estimator.run(data, model_class, seed = test_id)
                if model is not None:
                    np.savetxt(f'{batch_save_path}/results/{estimator_name}/test_{test_id}_inliers.txt', inliers)
                    np.savetxt(f'{batch_save_path}/results/{estimator_name}/test_{test_id}_params.txt', model.params)
                    np.savetxt(f'{batch_save_path}/results/{estimator_name}/test_{test_id}_iterations.txt', np.array([iterations]))
                    np.savetxt(f'{batch_save_path}/results/{estimator_name}/test_{test_id}_scores.txt', scores)
                    model_residuals = model.residuals(data)
                    np.savetxt(f'{batch_save_path}/results/{estimator_name}/test_{test_id}_residuals.txt', model_residuals)

        with finished_tests.get_lock():
            finished_tests.value += 1

        percentage_completed = "%.2f" % float(finished_tests.value / n_tests * 100)
        sys.stdout.write(f'\rEstimating model parameters for {batch_id} with {n_tests} tests | {percentage_completed} % Complete')
        sys.stdout.flush()

        return True

    #for test_id in range(n_tests):
    #    _run_test(test_id)

    # run estimation tests
    finished_tests = Value('i', 0)

    def _init_pool(args):
        global finished_tests
        finished_tests = args

    with Pool(processes = 8, initializer = _init_pool, initargs = (finished_tests, )) as p:
        p.map(_run_test, range(n_tests))

    print(' ')