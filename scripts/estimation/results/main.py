from estimation.fit import EllipseModel, PlaneModelND, HomographyModel
from estimation.results.utils import _get_estimation_error, _angle_between_vectors, _compute_RMSE
from test_configuration.utils import _read_yaml

import numpy as np
import sys, os

if __name__ == '__main__':

    model_class = str(sys.argv[1])
    batch_id = str(sys.argv[2])

    # read batch params
    current_path = os.path.dirname(os.path.realpath(__file__))
    scripts_path = current_path[:-19]
    tests_path = _read_yaml(f'{scripts_path}/test_configuration/params/tests_path.yaml')['path']
    batch_save_path = f'{tests_path}/{model_class}/{batch_id}'
    batch_params = _read_yaml(f'{batch_save_path}/batch_params.yaml')    
    
    estimators_names = batch_params['estimators_names']
    n_tests = batch_params['n_tests']
    model = eval(batch_params['model_params']['model_class'])()
    estimators_names = batch_params['estimators_names']
    n_estimators = len(estimators_names)
    
    # counter of estimators whose tests results have been generated
    finished_estimators = 1

    for estimator in estimators_names:
    
        # counter tests whose results have been generated
        finished_tests = 0

        n_successful_tests = n_tests # at first we assume all tests have been successfully estimated

        # Errors lists
        estimation_errors = [] # estimation errors, depends on model_class
        abs_errors = []     # absolute and relative errors for each component of the parameters vector
        rel_errors = []	
        iterations_list = []     # iterations of the iterative reestimation stage
            
        for test_id in range(n_tests):

            try:
                # model original and estimated params
                test_params = _read_yaml(f'{batch_save_path}/tests_params/test_{test_id}.yaml')
                params_original = test_params['model_params']['params']
                params_estimated = np.loadtxt(f'{batch_save_path}/results/{estimator}/test_{test_id}_params.txt')
                if estimator != 'default':
                    iterations = np.loadtxt(f'{batch_save_path}/results/{estimator}/test_{test_id}_iterations.txt')

                if model.__class__ == HomographyModel:
                    # read dataset and original inliers to compute RMSE
                    data1 = np.loadtxt(f'{batch_save_path}/datasets/test_{test_id}_proj1.txt')
                    data2 = np.loadtxt(f'{batch_save_path}/datasets/test_{test_id}_proj2.txt')
                    data = np.column_stack((data1,data2))
                    original_inliers = np.loadtxt(f'{batch_save_path}/datasets/test_{test_id}_inliers.txt').astype(bool)

            except IOError:
                n_successful_tests -= 1
                print(f'    Test {str(id + 1)} is not considered in results')
                continue
            except ValueError:
                n_successful_tests -= 1
                print(f'    Cannot read ../{estimator}/test_{id}_params.txt')
                continue

            # convert params to general coefficients and compute estimation_error
            if model.__class__ == PlaneModelND:
                params_original = np.asarray(params_original).reshape(2, -1)
                estimation_error, is_antiparallel = _angle_between_vectors( params_original[1], 
                                                                            params_estimated[1])
                if is_antiparallel:
                    params_estimated[1] = -params_estimated[1]

            params_original = model.get_general_params(params_original)
            params_estimated = model.get_general_params(params_estimated)

            if model.__class__ == EllipseModel:
                _, rel_error = _get_estimation_error(params_original, params_estimated)
                estimation_error = np.max(rel_error)
            
            if model.__class__ == HomographyModel:
                estimation_error = _compute_RMSE(params_original, params_estimated, data, original_inliers)
            
            estimation_errors = np.append(estimation_errors, estimation_error)

            # abs and rel errors
            abs_error, rel_error = _get_estimation_error(params_original, params_estimated)
            abs_errors = np.append(abs_errors, abs_error)
            rel_errors = np.append(rel_errors, rel_error)
            # some additional info
            if estimator != 'default': iterations_list = np.append(iterations_list, iterations)
            
            finished_tests += 1

            percentage_completed = "%.2f" % float(finished_tests / n_tests * 100)
            sys.stdout.write(f'\rGenerating results for {batch_id} with {n_tests} tests | {finished_estimators}/{n_estimators} Completed | {percentage_completed} % Complete')
            sys.stdout.flush()
    
        # reshape and save
        estimation_errors = estimation_errors.reshape(n_successful_tests, -1)
        abs_errors = abs_errors.reshape(n_successful_tests, -1)
        rel_errors = rel_errors.reshape(n_successful_tests, -1)
        if estimator != 'default': iterations_list = iterations_list.reshape(n_successful_tests, -1)

        np.savetxt(f'{batch_save_path}/results/{estimator}/00_estimation_errors.txt', estimation_errors)
        np.savetxt(f'{batch_save_path}/results/{estimator}/00_abs_errors.txt', abs_errors)
        np.savetxt(f'{batch_save_path}/results/{estimator}/00_rel_errors.txt', rel_errors)
        if estimator != 'default': np.savetxt(f'{batch_save_path}/results/{estimator}/00_iterations.txt', iterations_list)

        finished_estimators += 1
    
    print(' ')