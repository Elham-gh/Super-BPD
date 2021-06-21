from datasets import FluxSegmentationDataset
from post_process import generate_cluster
import argparse
import os
import torch
import torch.nn as nn
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle


DATASET = 'nyu'
TEST_VIS_DIR = './test_CEN_nyu_pred_flux/'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Super-BPD Network")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="Dataset for training.")
    parser.add_argument("--test-vis-dir", type=str, default=TEST_VIS_DIR,
                        help="Directory for saving vis results during testing.")
    return parser.parse_args()

args = get_arguments()

def main():
    
    with open('/content/SuperBPD/data/nyu/val.txt', 'r') as f:
        # names = f.readlines()
        image_names = f.read().splitlines()
        names = [name[4:] for name in image_names]

    # with open('/content/SuperBPD/bpds.txt', 'w') as f:
    a, b = {}, {}
    for i_iter, image_name in enumerate(names):

        print(i_iter, image_name)
        path = 'test_CEN_nyu_pred_flux/nyu/' + image_name + '.mat'
        
        out = generate_cluster.main(path, True)
        b[image_name] = out['before']
        a[image_name] = out['after']
            
    after = open('./CEN_bpds.pkl', 'wb')
    pickle.dump(a, after)

    before = open('./CEN_bpds_b.pkl', 'wb')
    pickle.dump(b, before)
    
        
if __name__ == '__main__':
    main()
