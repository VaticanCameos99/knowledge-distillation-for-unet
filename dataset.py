import torch
from torch.utils.data import Dataset
from data_augmentation import load_data, split_squares
import random
import numpy as np

class listDataset(Dataset):

    def __init__(self, ids, shuffle = True, transform = None, num_workers = 4):
        #data loading
        random.shuffle(ids)

        self.nSamples = len(ids)
        self.lines = ids
        self.transform = transform
        #self.batch_size = batch_size -> Bs is mentioned in dataloader
        self.num_workers = num_workers
         

    def __getitem__(self, index):
        assert index <= len(self), 'Error: index out of bound'
        
        img_path = self.lines[index]
        img, gt = load_data(img_path)

        if self.transform is not None:
            img = self.transform(img)

        #img = np.array(img)
        gt = np.array(gt)
        #split to squares
        gt = np.expand_dims(gt, axis = 2)
        gt = gt.transpose(2, 0, 1)

        '''img_left = split_squares(img, 0)
        img_right = split_squares(img, 1)

        gt_left = split_squares(gt, 0)
        gt_right = split_squares(gt, 1)

        print(img.shape, 'img shape after tf')
        print(gt.shape, 'gt shape after tf')

        img_list.append(img_left)        
        gt_list.append(gt_left)
        img_list.append(img_right)        
        gt_list.append(gt_right)
        #different sizes
        print(img_left.shape, 'img left')
        print(gt_left.shape, 'gt left')
        print(img_right.shape, 'img right')

    
        img_list = np.array(img_list)
        gt_list = np.array(gt_list)

'''
        #returning imgs and gts as a tensor
        #(1, 3, 320, 320), (1, 1, 320, 320)  - for batch_size = 1
        gt = np.array(gt)
        i = img.float()
        g = torch.from_numpy(gt).float()
        return i, g
            

            

    def __len__(self):
        #len(dataset)
        return self.nSamples