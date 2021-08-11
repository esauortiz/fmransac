from test_configuration.utils import _read_yaml
from estimation.fit import PlaneModelND, EllipseModel, HomographyModel
from estimation.fit import CircleModel, LineModelND
from dataset_generation.utils import _uniform_noise, _get_bbox_from_value
import numpy as np
import sys, os

if __name__ == '__main__':

    model_class = str(sys.argv[1])
    batch_id = str(sys.argv[2])

    current_path = os.path.dirname(os.path.realpath(__file__))
    scripts_path = current_path[:-19]
    yaml_file = f'{scripts_path}/test_configuration/params/batch_group.yaml'
    batch_group_params = _read_yaml(yaml_file)
    save_path = batch_group_params['group_params']['save_path']
    initial_batch_id = batch_group_params['group_params']['save_path']

    batch_params = _read_yaml(f'{save_path}/{model_class}/{batch_id}/batch_params.yaml')
    dataset_params = batch_params['dataset_params']
    save_path = batch_params['save_path']
    n_tests = batch_params['n_tests']
    model_class = batch_params['model_params']['model_class']

    for test_id in range(n_tests):

        test_params = _read_yaml(f'{save_path}/tests_params/test_{test_id}.yaml')
        model_params = test_params['model_params']

        # generate original model data
        if model_class == 'PlaneModelND':
            params = model_params['params']
            original_model = PlaneModelND(params)
            dim = len(params[0]) # dim = number of components of origin
            model_samples = model_params['model_samples']
            model_bbox = _get_bbox_from_value(dataset_params['model_bbox'], dim)
            original_data = original_model.predict(model_bbox, model_samples, seed = test_id)
            from dataset_generation.PlaneModelND import _gaussian_noise

        elif model_class == 'EllipseModel':
            params = model_params['params']
            original_model = EllipseModel(params)
            model_samples = model_params['model_samples']
            model_bbox = _get_bbox_from_value(dataset_params['model_bbox'], dim = 2)
            original_data = original_model.predict_xy(t = np.linspace(0, 2 * np.pi, model_samples), model_bbox = model_bbox)
            from dataset_generation.EllipseModel import _gaussian_noise

        elif model_class == 'CircleModel':
            params = model_params['params']
            original_model = CircleModel(params)
            model_samples = model_params['model_samples']
            #model_bbox = _get_bbox_from_value(dataset_params['model_bbox'], dim = 2)
            original_data = original_model.predict_xy(t = np.linspace(0, 2 * np.pi, model_samples))
            #from dataset_generation.CircleModel import _gaussian_noise

        elif model_class == 'LineModelND':
            params = model_params['params']
            original_model = LineModelND(params)
            model_samples = model_params['model_samples']
            #model_bbox = _get_bbox_from_value(dataset_params['model_bbox'], dim = 2)
            #original_data = original_model.predict(TODO)
            #from dataset_generation.CircleModel import _gaussian_noise

        elif model_class == 'HomographyModel':
            original_model = HomographyModel

        if model_class in ['PlaneModelND', 'EllipseModel']:
            try:
                inliers = _gaussian_noise(original_data, original_model, dataset_params, seed = test_id)
                outliers = _uniform_noise(original_data, original_model, dataset_params, seed = test_id)
                noisy_data = np.concatenate((inliers, outliers), axis = 0)

                # write noisy data file
                np.savetxt((test_params['save_path'] + '/datasets/test_' + str(test_id) + '.txt'), noisy_data)
                # write original data file (for plotting pourposes)
                np.savetxt((test_params['save_path'] + '/datasets/test_' + str(test_id) + '_original.txt'), noisy_data)
            
            except RuntimeError:
                print(f'Data for test {test_id} has not been generated')