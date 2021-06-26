import torch
import bpd_cuda
import math
import scipy.io as sio
import cv2
import numpy as np
from matplotlib import pyplot as plt

def label2color(label):

    label = label.astype(np.uint16)
    
    height, width = label.shape
    color3u = np.zeros((height, width, 3), dtype=np.uint8)
    unique_labels = np.unique(label)

    if unique_labels[-1] >= 2**24:       
        raise RuntimeError('Error: label overflow!')

    for i in range(len(unique_labels)):
    
        binary = '{:024b}'.format(unique_labels[i])
        # r g b 3*8 24
        r = int(binary[::3][::-1], 2)
        g = int(binary[1::3][::-1], 2)
        b = int(binary[2::3][::-1], 2)

        color3u[label == unique_labels[i]] = np.array([r, g, b])

    return color3u

def write(results, image_name):
  
    root_points, super_BPDs_before_dilation, super_BPDs_after_dilation, super_BPDs = results

    root_points = root_points.cpu().numpy()
    super_BPDs_before_dilation = super_BPDs_before_dilation.cpu().numpy()
    super_BPDs_after_dilation = super_BPDs_after_dilation.cpu().numpy()
    super_BPDs = super_BPDs.cpu().numpy()

    # cv2.imwrite('root.png', 255*(root_points > 0))
    # cv2.imwrite('super_BPDs.png', label2color(super_BPDs))
    # cv2.imwrite('super_BPDs_before_dilation.png', label2color(super_BPDs_before_dilation))
    # cv2.imwrite('super_BPDs_after_dilation.png', label2color(super_BPDs_after_dilation))

    fig = plt.figure(figsize=(10,6))

    ###* ax0        input image
    ax0 = fig.add_subplot(221)
    ax0.imshow(255*(root_points > 0))#(vis_image[:,:,::-1])
    ax0.set_title('root')

    ax1 = fig.add_subplot(222)
    ax1.set_title('SuperBPD')
    ax1.set_autoscale_on(True)
    im1 = ax1.imshow(label2color(super_BPDs))
    # plt.colorbar(im1,shrink=0.5)

    ###* ax2        
    ax2 = fig.add_subplot(223)
    ax2.set_title('before')
    ax2.set_autoscale_on(True)
    im2 = ax2.imshow(label2color(super_BPDs_before_dilation))
    # plt.colorbar(im2, shrink=0.5)

    ###* ax2        
    ax2 = fig.add_subplot(224)
    ax2.set_title('after')
    ax2.set_autoscale_on(True)
    im2 = ax2.imshow(label2color(super_BPDs_after_dilation))
    # plt.colorbar(im2, shrink=0.5)

    plt.savefig('./output/' + image_name + '.png')
    plt.close(fig)
    
    return results


###* a function to return labeled results
#def get_output(super_BPDs_before_dilation, super_BPDs_after_dilation):
    
#    return {'before': super_BPDs_before_dilation, 'after': super_BPDs_after_dilation}
    
  
def main(path='./2009_004607.mat', writing=True):
    
    flux = sio.loadmat(path)['flux']
    flux = torch.from_numpy(flux).cuda()

    angles = torch.atan2(flux[1,...], flux[0,...]).contiguous()
    angles[angles < 0] += 2*math.pi

    height, width = angles.shape

    # unit: degree
    # theta_a, theta_l, theta_s, S_o, 45, 116, 68, 5
    ###* results includes final outputs
    results = bpd_cuda.forward(angles, height, width, 45, 116, 68, 5)
    root_points, super_BPDs_before_dilation, super_BPDs_after_dilation, super_BPDs = results
    
    super_BPDs_before_dilation = super_BPDs_before_dilation.cpu().numpy()
    super_BPDs_after_dilation = super_BPDs_after_dilation.cpu().numpy()

    if writing:
        write(results, path[-10:-4])
        
    return {'before': super_BPDs_before_dilation, 'after': super_BPDs_after_dilation}
    
