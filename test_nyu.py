import argparse
import os
import torch
import torch.nn as nn
from model import VGG16, ResNetLW, Bottleneck
from vis_flux import vis_flux
from dataset_nyu import FluxSegmentationDataset
from torch.autograd import Variable
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader

DATASET = 'nyu'
TEST_VIS_DIR = './test_CEN_nyu_pred_flux/'
SNAPSHOT_DIR = './snapshots/'
SAVED_MODEL = './saved/'

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
    parser.add_argument("--saved-model", type=str, default=SAVED_MODEL,
                        help="saved model for generating output")                        
                        
    return parser.parse_args()

args = get_arguments()

def main():

    if not os.path.exists(args.test_vis_dir + args.dataset):
        os.makedirs(args.test_vis_dir + args.dataset)

    model = ResNetLW(Bottleneck, [3, 4, 6, 3])

    model.load_state_dict(torch.load(args.saved_model + 'PascalContext_400000.pth'))

    model.eval()
    model.cuda()
    
    dataloader = DataLoader(FluxSegmentationDataset(dataset=args.dataset, mode='val'), batch_size=1, shuffle=False, num_workers=4)

    for i_iter, batch_data in enumerate(dataloader):

        Input_image, vis_image, gt_mask, gt_flux, weight_matrix, dataset_lendth, image_name = batch_data
        ###* input_image        normalized, RGB input image
        ###* vis_image          original input image
        ###* gt_mask            label + 1 (no zero class)
        ###* gt_flux            direction_field
        ###* weight_matrix      inversely proportional to # labels of a class
        ###* dataset_length     # images
        ###* image_name         

        print(i_iter, dataset_lendth)

        ###* pred_flux          output image of model performed on input image
        pred_flux = model(Input_image.cuda())

        vis_flux(vis_image, pred_flux, gt_flux, gt_mask, image_name, args.test_vis_dir + args.dataset + '/')

        pred_flux = pred_flux.data.cpu().numpy()[0, ...]
        sio.savemat(args.test_vis_dir + args.dataset + '/' + image_name[0] + '.mat', {'flux': pred_flux})


if __name__ == '__main__':
    main()
