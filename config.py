import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/orca_tpl/'
MODEL_DIR = os.path.join(PROJECT_ROOT_DIR, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, 'output')

print(PROJECT_ROOT_DIR)
