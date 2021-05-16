from datasets import FluxSegmentationDataset
from post_process import generate_cluster
import argparse
import os
import torch
import torch.nn as nn
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import numpy as np


DATASET = 'nyu'
TEST_VIS_DIR = './test_nyu_pred_flux/'

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
    
    with open('/content/SuperBPD/nyu/val/val.txt', 'r') as f:
        names = f.readlines()

    with open('/content/SuperBPD/bpds.txt', 'w') as f:

        for i_iter, image_name in enumerate(names):

            print(i_iter, image_name)
            path = 'test_nyu_pred_flux/nyu/' + image_name[:-1] + '.mat'

            out = generate_cluster.main(path, True)
            b = out['before']
            a = out['after']
            # print(np.unique(b).shape)#=(5335,)
            # print(np.unique(a).shape)#=(403,)
            f.write(image_name[:-1] + '\n')
            f.write(str(a))
            f.write('\n')


        
if __name__ == '__main__':
    main()
