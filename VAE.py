import torch.nn as nn
import torch


class VAE(nn.Module):

    def __init__(self,latentDim,device):
        super(VAE,self).__init__()
        self.device = device
        self.latentDim = latentDim
        self.Encoder = VAE_Encoder(self.latentDim)
        self.Decoder = VAE_Decoder(self.latentDim)
        
    
    def forward(self,X):
        u , log_s = self.Encoder(X)
        e = torch.randn((X.shape[0],self.latentDim)).to(self.device)
        Z = u + e * torch.exp(log_s)
        X = self.Decoder(Z)
        return X , u, log_s




class VAE_Encoder(nn.Module):
    # for this Test no full covariance matrix is used for the guassian
    def __init__(self,latentDim):
        super(VAE_Encoder,self).__init__()

        
        self.M1 = nn.Sequential(
            nn.Conv2d(3,32,5,2),
            nn.ReLU(),
            nn.Conv2d(32,64,3,2),
            nn.ReLU(),
            nn.Conv2d(64,128,2,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3200,1600),
            nn.ReLU(),

        )

        self.L1 = nn.Linear(1600,latentDim)
        self.L2 = nn.Linear(1600,latentDim)
    def forward(self,X):
        X = self.M1(X)
        mean = self.L1(X)
        log_standardDeviation = self.L2(X)
        return  mean, log_standardDeviation
        
       

    

class VAE_Decoder(nn.Module):
    
    def __init__(self,latent_dim):
        super(VAE_Decoder,self).__init__()

        self.M1 = nn.Sequential(
            nn.Linear(latent_dim,4*4*1024),
            nn.Unflatten(1,(1024,4,4)),
            nn.ConvTranspose2d(1024,512,5,1),
            nn.ReLU(),
            nn.ConvTranspose2d(512,256,2,2),
            nn.ReLU(),
            nn.ConvTranspose2d(256,3,2,2),
            nn.Tanh(),
        )
    
    def forward(self,X):
        return self.M1(X)