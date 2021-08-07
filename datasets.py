import cv2
import numpy as np
import os.path as osp
import scipy.io as sio
from torch.utils.data import Dataset

IMAGE_MEAN = np.array([103.939, 116.779, 123.675], dtype=np.float32)

class FluxSegmentationDataset(Dataset):

    def __init__(self, dataset='sun', mode='train'):
        
        self.dataset = dataset
        self.mode = mode

        # file_dir = self.dataset + '/' + self.mode + '/' + self.mode + '.txt'
        file_dir = '/content/drive/MyDrive/datasets/sunrgbd/train/train.txt'

        self.random_flip = False
        
        if self.dataset == 'PascalContext' and mode == 'train':
            self.random_flip = True

        with open(file_dir, 'r') as f:
            names = f.read().splitlines()
            self.image_names = [name[4:] for name in names]


        self.dataset_length = len(self.image_names)
    
    def __len__(self):

        return self.dataset_length

    def __getitem__(self, index):

        random_int = np.random.randint(0,2)

        image_name = self.image_names[index]

        image_path = osp.join('/content/drive/MyDrive/datasets/sunrgbd/train/images', 'img-' + image_name + '.png')
        
        image = cv2.imread(image_path, 1)
        
        if self.random_flip:
            if random_int:
                image = cv2.flip(image, 1)
        
        vis_image = image.copy()

        height, width = image.shape[:2]
        image = image.astype(np.float32)
        image -= IMAGE_MEAN
        image = image.transpose(2, 0, 1)

        if self.dataset == 'PascalContext':
            label_path = osp.join('datasets', self.dataset, 'labels', image_name + '.mat')
            label = sio.loadmat(label_path)['LabelMap']
        
        elif self.dataset == 'BSDS500':
            label_path = osp.join('datasets', self.dataset, 'labels', image_name + '.png')
            label = cv2.imread(label_path, 0)
        
        elif self.dataset == 'nyu':
            label_path = osp.join('/content/drive/MyDrive/datasets/nyudv2/masks', image_name[-6:] + '.png')
            label = cv2.imread(label_path, 0)

        elif self.dataset == 'sun':
            label_path = '/content/drive/MyDrive/datasets/sunrgbd/train/labels/img-' + image_name + '.png'
            
            print(label_path)
            label = cv2.imread(label_path, 0)

        if self.random_flip:
            if random_int:
                label = cv2.flip(label, 1)
        
        ###* Prevention of 0 label
        label += 1
        
        gt_mask = label.astype(np.float32)
        
        ###* All existing labels
        categories = np.unique(label)

        if 0 in categories:
            raise RuntimeError('invalid category')
        
        ###* Label image zero padding
        label = cv2.copyMakeBorder(label, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        
        ###* equal sizes to label
        weight_matrix = np.zeros((height+2, width+2), dtype=np.float32)
        ###* Two channels for 2-dim vectors
        direction_field = np.zeros((2, height+2, width+2), dtype=np.float32)

        for category in categories:
            ###* 0-1 masked GT image corresponding to the category
            img = (label == category).astype(np.uint8)
            ###* A unique weight matrix for all categories, propertionate to reverse of number of pixels of category
            weight_matrix[img > 0] = 1. / np.sqrt(img.sum())
            
            ###* Making GT for BPDs, using a written function of opencv
            ###* Boundaries of img are boundary of the complete objects of each class not all edges withing the picture
            ###* _ = distance, labels = zero pixels
            _, labels = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)

            ###* index = zeros to which a foreground pixels is closest, place = indices of zeros ### supposed vice vera
            index = np.copy(labels)
            index[img > 0] = 0
            place =  np.argwhere(index > 0)
            
            nearCord = place[labels-1,:]
            x = nearCord[:, :, 0]
            y = nearCord[:, :, 1]
            nearPixel = np.zeros((2, height+2, width+2))
            nearPixel[0,:,:] = x
            nearPixel[1,:,:] = y
            grid = np.indices(img.shape)
            grid = grid.astype(float)
            diff = grid - nearPixel    

            direction_field[:, img > 0] = diff[:, img > 0]     

        weight_matrix = weight_matrix[1:-1, 1:-1]
        direction_field = direction_field[:, 1:-1, 1:-1]
        
        if self.dataset == 'BSDS500':
            image_name = image_name.split('/')[-1]

        return image, vis_image, gt_mask, direction_field, weight_matrix, self.dataset_length, image_name

    
