import os
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

class UnalignedDataset(data.Dataset):
    def __init__(self):
        self.l_path = './datasets/scenery/trainA/'
        self.p_path = './datasets/scenery/trainB/'
        
        self.l_imgs = []
        for root, _, names in sorted(os.walk(self.l_path)):
            for name in names:
                self.l_imgs.append(os.path.join(root, name))
        self.p_imgs = []
        for root, _, names in sorted(os.walk(self.p_path)):
            for name in names:
                self.p_imgs.append(os.path.join(root, name))
        self.l_size = len(self.l_imgs)
        self.p_size = len(self.p_imgs)
        
        self.img_loader = transforms.Compose([
            transforms.Resize([286,286],transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    
    def __len__(self):
        return max(self.l_size, self.p_size)
    
    def __getitem__(self, index):
        l_path = self.l_imgs[index % self.l_size]
        p_path = self.p_imgs[index % self.p_size]
        l_img = Image.open(l_path).convert('RGB')
        p_img = Image.open(p_path).convert('RGB')
        l_img = self.img_loader(l_img)
        p_img = self.img_loader(p_img)
        return l_img, p_img