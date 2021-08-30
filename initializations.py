import torch
import torch.optim as opt 
import torchvision.datasets as dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import config as c 
from model import generator, discriminator



d = discriminator(c.img_dim).to(c.device) # create the objects of the model
g = generator(c.z, c.img_dim).to(c.device)
print(d)


fixed_noise = torch.randn((c.batch_size, c.z)).to(c.device) # initialization of the fixed noise to input the generator;
# it is fixed becuase we will see how it is changed after the learning across the epoch



transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))]
    ) # transformation for dataset



train_dataset = dataset.MNIST(root = 'dataset/', download = True, transform = transform) # Dataset creating by using pytorch dataset
train_loader =  DataLoader(train_dataset, batch_size = c.batch_size, shuffle = True)



optimizerg = opt.Adam(g.parameters(), lr=c.lr)  # Setup optmizers
optimizerd = opt.Adam(d.parameters(), lr=c.lr)



criterion = nn.BCELoss() # loss finction


writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0
# write this to tensorboard for better visualization
# This writer will out fake images the generator has generated




