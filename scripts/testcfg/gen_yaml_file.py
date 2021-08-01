import numpy as np
import sys
import random
import yaml
import io

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

# test meta params
test_header = 'test_'
test_batch = sys.argv[1]
test_num = int(sys.argv[2])
save_path = sys.argv[3]

# Read yaml master
yaml_file = save_path + '/batch_params.yaml'
with open(yaml_file, 'r') as stream:
    batch_params = yaml.safe_load(stream)

# Select only current batch params
model_class = batch_params['model_class']
data_params = batch_params['data_params']
noise_params = batch_params['noise_params']

for test_id in range(1, test_num + 1):
    # random seed
    # seed_mod setted at 0 as default
    random.seed(test_id)
    np.random.seed(test_id)

    # Model
    model_params = None
    data_len = None

    if model_class == 'LineModelND':
        axis = data_params['axis']
        axis_range = [- random.randint(*data_params['axis_range']), 
                        random.randint(*data_params['axis_range'])]

        quadrant = data_params['quadrant']
        if quadrant == 0:
            quadrant = None
        origin, direction = _get_origin_direction(  data_params['origin'], 
                                                    data_params['direction'], 
                                                    data_params['dim'],
                                                    quadrant)
        # the range will be applied to the direction component with most weight
        max_index_col = np.argmax(abs(np.array([*direction])), axis=0)
        data_len = [axis_range, int(max_index_col)]
        model_params = [origin, direction]

    elif model_class == 'CircleModel':

        xc = random.randint(*data_params['xc'])
        yc = random.randint(*data_params['yc'])
        radius = random.randint(*data_params['radius'])

        data_len = data_params['samples']
        model_params = [xc, yc, radius]

    elif model_class == 'EllipseModel':

        xc = random.randint(*data_params['xc'])
        yc = random.randint(*data_params['yc'])
        height = random.randint(*data_params['a'])
        width = random.randint(*data_params['b'])
        theta = random.randint(*data_params['theta'])
        theta = float(theta / 10**(3))
        
        data_len = data_params['samples']
        model_params = [xc, yc, height, width, theta]

    elif model_class == 'PlaneModelND':
        origin, direction = _get_origin_direction(data_params['orgn_range'], data_params['nrm_vctr_range'], (data_params['dim']))
        data_len = [data_params['ranges'], data_params['samples']]

        model_params = [origin, direction]

    elif model_class == 'HomographyModel':
        
        # 3D plane params
        origin, direction = _get_origin_direction(data_params['orgn_range'], data_params['nrm_vctr_range'], 3)
        # ranges and samples of 3D plane
        data_len = [data_params['ranges'], data_params['samples']]
        # Camera extrinsincs params
        theta = data_params['theta']
        tx = data_params['tx']
        ty = data_params['ty']
        tz = data_params['tz']

        model_params = [theta, tx, ty, tz, origin, direction]

    # Set file name
    file_name = test_header + str(test_id)

    # Define data
    payload = {
        'data_params' : {
            'model': model_class,
            'data_len' : data_len,
            'model_params': model_params,
            'gn_params': noise_params['gn_params'],
            'un_params': noise_params['un_params'],
            'on_params': noise_params['on_params'],
            'bbox_limits': data_params['bbox_limits'],
            'bbox_limits_tolerance' : data_params['bbox_limits_tolerance']
        },
        'seed': test_id,
        'ransac_params' : [*batch_params['ransac_params'], test_id],
        'save_path' : save_path,
        'file_name' : file_name
    }
    # Write YAML file
    with io.open((save_path + '/yaml/' + file_name + '.yaml'), 'w', encoding='utf8') as outfile:
        yaml.dump(payload, outfile, default_flow_style=False, allow_unicode=True)

