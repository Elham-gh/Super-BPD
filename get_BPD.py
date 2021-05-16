from datasets import FluxSegmentationDataset
from post_process import generate_cluster
import argparse
import os
import torch
import torch.nn as nn
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader


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
    
    dataloader = DataLoader(FluxSegmentationDataset(dataset=args.dataset, mode='test'), batch_size=1, shuffle=False, num_workers=4)
    
    for i_iter, batch_data in enumerate(dataloader):

        _, _, _, _, _, _, image_name = batch_data

        print(i_iter, image_name[0])
        path = 'test_pred_flux/PascalContext/' + image_name[0] + '.mat'

        before, after = generate_cluster.main(path, True)
        print('***********************************/n', before)
        print('***********************************/n', after)
        hi
        
if __name__ == '__main__':
    main()
