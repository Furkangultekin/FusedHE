from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import cv2

#import pandas as pd
#from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
#from osgeo import gdal
from torchvision import transforms as T
import albumentations as A


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class LocalDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=False,inference=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        if transform:
            self.transform = [
                    A.HorizontalFlip(),
                   # A.RandomCrop(crop_size[0], crop_size[1]),
                    A.RandomBrightnessContrast(),
                    A.RandomGamma(),
                    A.HueSaturationValue(),
                    A.Resize(512,512)
                            #T.Normalize([0.3435, 0.3664, 0.3371], [0.1109, 0.1112, 0.1151])
                            #NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ]
        self.to_tensor = T.ToTensor()
        self.inference = inference

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir,"rgb/")))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_dir = os.path.join(self.root_dir,"rgb/")
        img_name = os.path.join(img_dir, os.listdir(img_dir)[idx])
        image = cv2.imread(img_name, cv2.COLOR_BGR2RGB)

        upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        if self.inference:
            image = self.to_tensor(image)
            nan_in = torch.isnan(image)
            image[nan_in] = 0.0001

            inf_in = torch.isinf(image)
            image[inf_in] = 0.0001
            data = {"image": image,
                    #"geo": {"xsize":xsize,
                    #        "ysize":ysize,
                    #        "geo_trans":geo_trans,
                    #        "projection":projection},
                    "fname": img_name      
                    }
        else:
            gt_dir = os.path.join(self.root_dir,"dsm/")
            gt_name = os.path.join(gt_dir,os.listdir(gt_dir)[idx])
            depth = cv2.imread(gt_name, cv2.IMREAD_UNCHANGED)# .astype('float32')
            segment = np.empty(depth.shape)

            additional_targets = {'depth': 'mask'}
            
            image = self.to_tensor(image)
            depth = self.to_tensor(depth)# .squeeze()
            segment = np.ones(depth.shape,dtype=np.float32)
            segment = torch.tensor(segment)

            nan_ind = torch.isnan(depth)
            depth[nan_ind] = 0.0001
            inf_in = torch.isinf(depth)
            depth[inf_in] = 0.0001 
            gt_ind = depth<=0.9
            depth[gt_ind] = 0.0001

            segment[gt_ind] =0 

            data = {"image": image,
                    "depth": depth/185,
                    "segment": segment,
                    #"geo": {"xsize":xsize,
                    #        "ysize":ysize,
                    #        "geo_trans":geo_trans,
                    #        "projection":projection},
                    "fname": img_name      
                    }

        return data

def normalize(arr):
    arr = ((arr)/(arr.max()))
    return arr