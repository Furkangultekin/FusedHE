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
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class InferenceDataset(Dataset):

    def __init__(self, root_dir, args):
        """
        Args:
            root_dir (string): Directory with all the inference images.
        """
        self.max_height = args.max_height_eval
        self.min_height = args.min_height_eval
        self.root_dir = root_dir
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir)))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_dir = os.path.join(self.root_dir)
        img_name = os.path.join(img_dir, os.listdir(img_dir)[idx])
        image = cv2.imread(img_name, cv2.COLOR_BGR2RGB)


        image = self.to_tensor(image)
        nan_in = torch.isnan(image)
        image[nan_in] = self.min_height

        inf_in = torch.isinf(image)
        image[inf_in] = self.min_height
        data = {"image": image,
                #"geo": {"xsize":xsize,
                #        "ysize":ysize,
                #        "geo_trans":geo_trans,
                #        "projection":projection},
                "fname": os.listdir(img_dir)[idx]      
                }
    
        return data

def normalize(arr):
    arr = ((arr)/(arr.max()))
    return arr