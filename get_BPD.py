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


DATASET = 'sun'
TEST_VIS_DIR = './train_sun_pred_flux/'

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
    
    after = dict()
    before = dict()

    with open('/content/drive/MyDrive/datasets/sunrgbd/train/train.txt', 'r') as f:
        names = f.readlines()

        for i_iter, image_name in enumerate(names):

            print(i_iter, image_name)
            path = '/content/SuperBPD/train_sun_pred_flux/sun/' + image_name[4:-1] + '.mat'

            out = generate_cluster.main(path, True)
            before[image_name[4:-1]] = out['before']
            after[image_name[4:-1]] = out['after']

    with open('/content/drive/MyDrive/datasets/sunrgbd/sun_train_bpds.pickle', 'wb') as f:
        pickle.dump(after, f)

    with open('/content/drive/MyDrive/datasets/sunrgbd/sun_train_bpds_b.pickle', 'wb') as f:
        pickle.dump(before, f)

        
if __name__ == '__main__':
    main()
