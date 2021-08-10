import yaml
import os
import shutil

def _read_yaml(file_name):
	with open(file_name, 'r') as stream: 
		return yaml.safe_load(stream)

def _create_directory(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def _remove_directory(dir):
	try:
		shutil.rmtree(dir)
	except OSError as e:
		pass
	
if __name__ == '__main__':
	True