# I am going to implement a GAN model for cifar10 dataset 
import torch
import torch.nn as nn


class generator(nn.Module):
    def __init__(self, z, img_dim):
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(z, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh() # to makesure that values should be between -1 and 1 becuase we are going to normalize input between -1 and 1      
            
        )
        
    def forward(self, x):
        return self.g(x)

class discriminator(nn.Module):
    def __init__(self, img_dim): # img_dim is 784 (28*28) because we are generating MNIST dataset; Flatten it will shows 784 dim
        super().__init__()
        self.d = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),     
            
        )
        
    def forward(self, x):
        return self.d(x)
        