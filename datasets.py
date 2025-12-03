from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import torch
import math
import random
from PIL import Image
import os
import glob
import einops
import torchvision.transforms.functional as F

class StainDataset(torch.utils.data.Dataset):
    def __init__(self, path, cls_number, center=None):
        self.path = path
        self.center = center
        self.five_path = ['../dataset/center_1_stain/',
                        '../dataset/center_2_stain/',
                        '../dataset/center_3_stain/',
                        '../dataset/center_4_stain/',
                        '../dataset/center_5_stain/']
        
        self.paths_counts = [sum([len(files) for _, _, files in os.walk(dir_path)]) for dir_path in self.five_path]
        if self.path=="five_centers":
            path = self.five_path[center]
            files = os.listdir(path)
            self.image_paths = [os.path.join(path, file) for file in files if file.endswith('.npy')]
        else:
            self.image_paths = []
            for path in self.five_path:
                files = os.listdir(path)
                npy_files = [os.path.join(path, file) for file in files if file.endswith('.npy')]
                self.image_paths.extend(npy_files)
        random.shuffle(self.image_paths)
        
        self.cls_number = cls_number
        self.probabilities_tensor = torch.tensor([count / sum(self.paths_counts) for count in self.paths_counts], dtype=torch.float32)
        self.cls = [0,1,2,3,4]
        self.data_len = len(self.image_paths)
        self.K = max(self.cls) + 1
        self.cnt = torch.tensor([len(np.where(np.array(self.cls) == k)[0]) for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / self.data_len for k in range(self.K)]
        
        print(f"How many data:{self.data_len}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx]).reshape((1, 2, 3))
        label = int(self.image_paths[idx].split('/')[-2].split('_')[1])-1
        return image, label
    
    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v
    
    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        return 1, 2, 3

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return "../dataset/npz/center_5_stain_v.npz"

    def sample_label(self, n_samples, device):
        return torch.full((n_samples,), self.cls_number, dtype=torch.int, device=device)

    def label_prob(self, k):
        return self.frac[k]

def center_path(center, split):
    metadata = pd.read_csv('data/metadata.csv')
    metadata = metadata.loc[metadata.loc[:, 'center'] == center,:]
    metadata = metadata.loc[metadata.loc[:, 'split'] == split,:]
    
    paths = []
    for idx in range(len(metadata)):
        data = metadata.iloc[idx,:]
        path = '../dataset/c17wilds/stains/patient_{0:03d}_node_{1}/patch_patient_{0:03d}_node_{1}_x_{2}_y_{3}.npy'.format(
            data.at['patient'], data.at['node'], data.at['x_coord'], data.at['y_coord'])
        if os.path.exists(path):
            paths.append(path)
    return paths

def get_label_list(images_list, label):
    return [label for _ in images_list]

def get_all_data_list(shuffle=True):
    all_images_list, all_labels_list = [], []
    
    for center in range(5):
        images_list = center_path(center, 0)
        labels_list = get_label_list(images_list, center)
        all_images_list.extend(images_list)
        all_labels_list.extend(labels_list)
    
    if shuffle:
        combined_paths = list(zip(all_images_list, all_labels_list))
        random.shuffle(combined_paths)
        return zip(*combined_paths)
    else:
        return all_images_list, all_labels_list

class C17WildsDataset(torch.utils.data.Dataset):
    def __init__(self, path, cls_number):
        self.path = path
        self.image_paths, self.label_paths = get_all_data_list(self.path)
        
        self.cls_number = cls_number
               
        print(f"How many data:{len(self.image_paths)}")
        print(f'{self.cls_number} classes')
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx]).reshape((1, 2, 3))
        label = self.label_paths[idx]
        return image, label
    
    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v
    
    def sample_label(self, n_samples, device):
        return torch.full((n_samples,), self.cls_number, dtype=torch.int, device=device)
    
    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        return 1, 2, 3

class C17EvalDataset(torch.utils.data.Dataset):
    def __init__(self, client):
        self.client = client
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return []
    
    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v
    
    def sample_label(self, n_samples, device):
        return torch.full((n_samples,), self.client, dtype=torch.int, device=device)
    
    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        return 1, 2, 3

def get_dataset(name, **kwargs):
    if name == 'c17wilds':
        return C17WildsDataset(**kwargs)
    elif name == 'stain':
        return StainDataset(**kwargs)
    elif name == 'StainEval':
        return C17EvalDataset(**kwargs)
    else:
        raise NotImplementedError(name)