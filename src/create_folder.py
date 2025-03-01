import os
import argparse
import yaml
from get_data import get_data, read_params  # Make sure this is correct

"""
    Creating a folder
"""

def create_folder(config_path, image=None):
    config = read_params(config_path)  # Read the YAML config properly
    dirr = config['load_data']['preprocessed_data']
    classes = config['load_data']['num_classes']

    if os.path.exists(os.path.join(dirr, 'train', 'class_0')) and os.path.exists(os.path.join(dirr, 'test', 'class_0')):
        print("--------Train and test folder already exists-------")
        print("-------Skipping the folder creation-------")
    else:
        os.makedirs(os.path.join(dirr, 'train'), exist_ok=True)
        os.makedirs(os.path.join(dirr, 'test'), exist_ok=True)
        for i in range(classes):
            os.makedirs(os.path.join(dirr, 'train', f'class_{i}'), exist_ok=True)
            os.makedirs(os.path.join(dirr, 'test', f'class_{i}'), exist_ok=True)
        print("--------Train and test folder created-------")

""" 
            Folder creation ended
"""

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="params.yaml")
    parsed_args = args.parse_args()
    
    create_folder(config_path=parsed_args.config)
