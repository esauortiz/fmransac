import os
import yaml

def _create_directory(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

if __name__ == "__main__":
    # read batch configuration
    yaml_file = 'batch_configuration.yaml'
    with open(yaml_file, 'r') as stream:
        batch_params = yaml.safe_load(stream)

    
    print(batch_params)
"""


	rm -r ${save_path}
	mkdir -p ${save_path}
	mkdir -p ${save_path}/data
	mkdir -p ${save_path}/results
	mkdir -p ${save_path}/results/default
	mkdir -p ${save_path}/results/ransac
	mkdir -p ${save_path}/results/msac
	mkdir -p ${save_path}/results/fun1
	mkdir -p ${save_path}/results/fun2
	mkdir -p ${save_path}/results/fun3
	mkdir -p ${save_path}/results/fun4
	mkdir -p ${save_path}/results/loransac
	mkdir -p ${save_path}/yaml
"""