import numpy as np
import random
import sys, os, io, yaml

from test_configuration.utils import _read_yaml

def _get_origin_direction(origin_range, direction_range, dim, quadrant = None):
    origin = []
    direction = []
    
    # For now selecting a quadrant is only available in 2D lines
    for i in range(dim):
        origin = [*origin, random.uniform(*origin_range)]
        dir_i = random.uniform(*direction_range)
        # in order not to get a direction parelel to an axis
        while dir_i == 0:
            dir_i = random.uniform(*direction_range)
        #direction = np.append(direction, dir_i)
        direction = [*direction, dir_i]

    if quadrant is not None:
        direction = [0, 0]
        # avoid lines paralel to an axis
        while 0 in direction:
            alpha = random.uniform(*quadrant)
            direction = [float(np.cos(np.deg2rad(alpha - 22.5))), float(np.sin(np.deg2rad(alpha - 22.5)))]

    return origin, direction

if __name__ == '__main__':
    
    print('[*] Setting tests parameters ...')

    model_class = sys.argv[1]
    group_id = sys.argv[2]

    # read batch group params
    current_path = os.path.dirname(os.path.realpath(__file__))
    tests_path = _read_yaml(f'{current_path}/params/tests_path.yaml')['path']
    batch_group_params = _read_yaml(f'{tests_path}/{model_class}/00_batch_groups/{group_id}/batch_group_params.yaml')
    n_batches = batch_group_params['group_params']['n_batches']
    initial_batch_id = batch_group_params['group_params']['initial_batch_id']
    n_tests = batch_group_params['group_params']['n_tests']
    model_class = batch_group_params['group_params']['model_class']

    for batch_id in range(n_batches):
        # read batch params
        batch_params = _read_yaml(f'{tests_path}/{model_class}/batch_{initial_batch_id + batch_id}/batch_params.yaml')
        model_params = batch_params['model_params']

        for test_id in range(n_tests):
            # random seed
            # seed_mod setted at 0 as default
            random.seed(test_id)
            np.random.seed(test_id)

            # Model
            test_model_params = None
            model_samples = None

            if model_class == 'LineModelND':
                axis = model_params['axis']
                axis_range = [- random.randint(*model_params['axis_range']), 
                                random.randint(*model_params['axis_range'])]

                quadrant = model_params['quadrant']
                if quadrant == 0:
                    quadrant = None
                origin, direction = _get_origin_direction(  model_params['origin'], 
                                                            model_params['direction'], 
                                                            model_params['dim'],
                                                            quadrant)
                # the range will be applied to the direction component with most weight
                max_index_col = np.argmax(abs(np.array([*direction])), axis=0)
                model_samples = [axis_range, int(max_index_col)]
                test_model_params = [origin, direction]

            elif model_class == 'CircleModel':

                xc = random.randint(*model_params['xc'])
                yc = random.randint(*model_params['yc'])
                radius = random.randint(*model_params['radius'])

                model_samples = model_params['samples']
                test_model_params = [xc, yc, radius]

            elif model_class == 'EllipseModel':

                xc = random.randint(*model_params['xc'])
                yc = random.randint(*model_params['yc'])
                height = random.randint(*model_params['a'])
                width = random.randint(*model_params['b'])
                theta = random.randint(*model_params['theta'])
                theta = float(theta / 10**(3))
                
                model_samples = model_params['samples']
                test_model_params = [xc, yc, height, width, theta]

            elif model_class == 'PlaneModelND':
                origin, direction = _get_origin_direction(model_params['origin_range'], model_params['normal_vector_range'], (model_params['dim']))
                model_samples = model_params['samples']

                test_model_params = [origin, direction]

            elif model_class == 'HomographyModel':
                
                # 3D plane params
                origin, direction = _get_origin_direction(model_params['orgn_range'], model_params['nrm_vctr_range'], 3)
                # ranges and samples of 3D plane
                model_samples = [model_params['ranges'], model_params['samples']]
                # Camera extrinsincs params
                theta = model_params['theta']
                tx = model_params['tx']
                ty = model_params['ty']
                tz = model_params['tz']

                test_model_params = [theta, tx, ty, tz, origin, direction]

            # Define data
            payload = {
                'test_id': test_id,
                'model_params' : {
                    'model': model_class,
                    'model_samples' : model_samples,
                    'params': test_model_params,
                },
                #'dataset_params' : batch_group_params['dataset_params'],
                #'ransac_params' : batch_group_params['ransac_params']
            }

            # Write YAML file
            file = f'{tests_path}/{model_class}/batch_{initial_batch_id + batch_id}/tests_params/test_{test_id}.yaml'
            with io.open(file, 'w', encoding='utf8') as outfile:
                yaml.dump(payload, outfile, default_flow_style=False, allow_unicode=True)