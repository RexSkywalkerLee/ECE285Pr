import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3, padding=0),
            nn.InstanceNorm2d(channel),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3, padding=0),
            nn.InstanceNorm2d(channel),
        )
    def forward(self, x):
        return x + self.conv_block(x)              
               
    
class Generator(nn.Module):
    def __init__(self, input_channel=3, output_channel=3, n_blocks=6):
        super(Generator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channel, 64, 7, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )
        
        self.res_blocks = nn.ModuleList([ResBlock(256) for i in range(n_blocks)])
            
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        
        self.conv6 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channel, 7, padding=0),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        return x
    
    
class Discriminator(nn.Module):
    def __init__(self, input_channel=3):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        
        self.conv5 = nn.Conv2d(512, 1, 4, stride=1, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        return x
    
    
class HED(nn.Module):
    def __init__(self):
        super(HED, self).__init__()
        
        self.vgg16 = models.vgg16(pretrained=True).features
        for param in self.vgg16.parameters():
            param.requires_grad = False
            
        self.score_dsn1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.score_dsn2 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        self.score_dsn3 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.score_dsn4 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        self.score_dsn5 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        
        self.upsample2 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample3 = nn.ConvTranspose2d(1, 1, kernel_size=6, stride=4, padding=1)
        self.upsample4 = nn.ConvTranspose2d(1, 1, kernel_size=12, stride=8, padding=2)
        self.upsample5 = nn.ConvTranspose2d(1, 1, kernel_size=24, stride=16, padding=4)
        
        self.fuse = nn.Conv2d(5, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        res = []
        for i, l in enumerate(self.vgg16):
            x = l(x)
            if i == 3:
                y = self.score_dsn1(x)
                res += [y]
            elif i == 8:
                y = self.score_dsn2(x)
                y = self.upsample2(y)
                res += [y]
            elif i == 15:
                y = self.score_dsn3(x)
                y = self.upsample3(y)
                res += [y]
            elif i == 22:
                y = self.score_dsn4(x)
                y = self.upsample4(y)
                res += [y]
            elif i == 29:
                y = self.score_dsn5(x)
                y = self.upsample5(y)
                res += [y]
                
        res = self.fuse(torch.cat(res, dim=1))
        return res