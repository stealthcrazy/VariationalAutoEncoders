import torch.nn as nn
import torch
import numpy as np
import json
import time
import datetime

from torch.utils.data import DataLoader

from os import listdir
from os.path import isfile, join

import VAE


## this is a cuda implementation

device = torch.device("cuda")



## copy pasted from Pytorch 
torch.backends.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR_DataLoader(torch.utils.data.Dataset):

    def __init__(self,dir,device):
        self.files = [f for f in listdir(dir) if isfile(join(dir, f))]
        self.Data = torch.tensor([])
        self.Labels = torch.tensor([])
        self.testData =torch.tensor([])
        self.testLabel =torch.tensor([])
        for file in self.files:
            #print(file)
            if "data" in file:
                tempData = unpickle(f"{dir}/{file}")

                self.Labels = torch.cat((self.Labels,torch.tensor(tempData[b"labels"])),0)

                self.Data = torch.cat((self.Data,torch.tensor(tempData[b"data"])),0)
    
            elif "test" in file:
                pass
        #print(self.Data.shape)
        
        self.Data = self.Data.reshape(self.Data.shape[0],3,32,32)
        self.Data = self.Data.to(torch.float32)
        #print(self.Data.shape)
        self.Data = (self.Data / 127.5) - 1
    def __len__(self):
        return self.Labels.shape[0]
        
    def __getitem__(self,index):
        return self.Data[index] , self.Labels[index]




batch_size = 128

Data = CIFAR_DataLoader("cifar-10-batches-py",device)
train_dataloader = DataLoader(Data, batch_size=batch_size, shuffle=True)

latentDim =32
Model = VAE.VAE(latentDim,device).to(device)
#Model = torch.compile(Model,mode="reduce-overhead")


criterion = nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(Model.parameters(), lr=1e-4, betas=(0.9, 0.99))

epochs = 1000

beta = 10

losses = []
kls = []
rls = []

for ep in range(epochs):

    for i, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        X = data[0].to(device)
        O , u , log_s = Model(X)
        kl = beta*((-0.5 * torch.sum(1 + (2*log_s) - (u**2) - torch.exp(2*log_s) ))/ X.size(0))
        rl = (criterion(O,X)/ X.size(0)) 
        loss = rl+kl
        
        loss.backward()
        optimizer.step()

       
        
        
        if i % 100 == 0:
            losses.append(loss.detach())
            rls.append(rl.detach())
            kls.append(kl.detach())
            info = f"Epoch {ep} : Step {i} \n: VAE_loss {loss.detach()} : KL div {kl.detach()} : Recon {rl.detach()} \n"
            with open("log.txt","a") as f:
                f.write(info)
            print("==========================")
            print(f"Epoch {ep} : Step {i} \n: VAE_loss {loss.detach()} : KL div {kl.detach()} : Recon {rl.detach()}")
            print("==========================")
    
    if ((ep % 50) == 0) and (ep != 0):
        ts = time.time()
        stmp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        torch.save({
            "Epoch":ep,
            "Model" : Model.state_dict(),
            "time"    : stmp,
            "LD" : latentDim,
            "Batch_Size" : batch_size,
            'Optimizer': optimizer.state_dict(),
            'MSE' : True,
            'ADAM' : True,
            'Losses':losses,
            'KL': kls,
            'Recon':rls,
            'Beta VAE': False
                    }, f'Checkpoint_Meta.pt')





ts = time.time()
stmp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
torch.save({
            "Epoch":ep,
            "Model" : Model.state_dict(),
            "time"    : stmp,
            "LD" : latentDim,
            "Batch_Size" : batch_size,
            'Optimizer': optimizer.state_dict(),
            'MSE' : True,
            'ADAM' : True,
            'Losses':losses,
            'KL': kls,
            'Recon':rls,
            'Beta VAE': False
                    }, f'Checkpoint_Meta.pt')



