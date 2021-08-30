# define hyperparameters 

import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"
# Note gans are sensitive to hyperparameters 
lr = 3e-4 # good learning rate for Adam optmizer 
z = 64 # you can also try 128, 256
img_dim = 28 * 28 * 1 # 784
batch_size = 32
epochs = 50 



