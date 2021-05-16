from datasets import FluxSegmentationDataset
from post_process import generate_cluster
import argparse
import os
import torch
import torch.nn as nn
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import numpy as np


DATASET = 'PascalContext'
TEST_VIS_DIR = './test_pred_flux/'
SNAPSHOT_DIR = './snapshots/'

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
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    return parser.parse_args()

args = get_arguments()

def main():
    
    with open('/content/SuperBPD/data/PascalContext/test.txt', 'r') as f:
        names = f.readlines()

    for i_iter, image_name in enumerate(names):

        print(i_iter, image_name)
        path = 'test_pred_flux/PascalContext/' + image_name[:-1] + '.mat'

        out = generate_cluster.main(path, True)
        b = out['before']
        a = out['after']
        # print('***********************************\n', b)
        # print('***********************************\n', a)
        #print(np.unique(b).shape)=(1534,)
        #print(np.unique(a).shape)=(90,)
        hi
        
if __name__ == '__main__':
    main()
