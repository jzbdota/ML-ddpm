"""
With Residual and Attention, out of memory in training!
"""


import torch
import torch.nn.functional as F
from torchvision import transforms as T, datasets
from torch.utils.data import DataLoader,Dataset,ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as timer
from datetime import datetime
from torch import nn,optim
from AttentionResidual import AttentionResidual

def imgTransform(img_size=64,reverse=False):
    if not reverse:
        transform = T.Compose([
            T.Resize((img_size,img_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(), # PIL to tensor, and rescale to [0,1]
            T.Lambda(lambda x: x*2 - 1) # rescale to [-1,1]
        ])
    else:
        transform = T.Compose([
            T.Lambda(lambda x: (x+1)/2), # rescale to [0,1]
            T.Lambda(lambda x: x.permute(1,2,0).clamp(min=0,max=1)), # CHW to HWC, for imshow
        ])
    return transform

def getdataloader(img_size=64,batch_size=128):
    """
    Get the dataloader of the stanford cars
    """
    transform = imgTransform(img_size=img_size,reverse=False)
    dset1 = datasets.StanfordCars("",transform = transform)
    dset2 = datasets.StanfordCars("",transform = transform,split='test')
    dset = ConcatDataset((dset1,dset2))
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

########################################################################################
class SinusoidalPosEmb(nn.Module):
    """
    Based on transformer-like embedding from 'Attention is all you need'
    Note: 10_000 corresponds to the maximum sequence length.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        import math
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class downblock(nn.Module):
    def __init__(self,in_channel,time_emb_dim,out_channel=None) -> None:
        super().__init__()
        if out_channel is None:
            out_channel = in_channel<<1

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(time_emb_dim,out_channel)
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,in_channel,kernel_size=4,stride=2,padding=1), # halve the image size
            block(in_channel,out_channel),
        )
        self.convnext = nn.Sequential(
            ResDoubleBlock(out_channel),
        )

    def forward(self,x,time_emb):
        time = self.mlp(time_emb)[...,None,None]
        h = self.conv(x)
        return self.convnext(h+time)
    
class upblock(nn.Module):
    def __init__(self,in_channel,time_emb_dim,out_channel=None) -> None:
        super().__init__()
        if out_channel is None:
            out_channel = in_channel>>1

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(time_emb_dim,out_channel)
        )

        self.conv0 = block(in_channel<<1,out_channel)
        self.conv = nn.Sequential(
            ResDoubleBlock(out_channel),
            nn.ConvTranspose2d(out_channel,out_channel,kernel_size=4,stride=2,padding=1)
        )

    def forward(self,x,time_emb):
        time = self.mlp(time_emb)[...,None,None]
        h = self.conv0(x)
        return self.conv(h + time)
###################################################################################

class ResDoubleBlock(nn.Module):
    def __init__(self,channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            block(channels),
            nn.Conv2d(channels,channels,3,padding=1,bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self,x):
        fx = self.conv(x)
        return F.relu(fx+x)

class block(nn.Module):
    def __init__(self,in_channel,out_channel=None) -> None:
        super().__init__()
        if not out_channel:
            out_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self,x):
        return self.conv(x)

class myUnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        total_channels = 5
        channels = [64<<i for i in range(total_channels)]
        time_emb_dim = channels[0]

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim,time_emb_dim<<2),
            nn.ReLU(),
            nn.Linear(time_emb_dim<<2,time_emb_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Sequential(
            block(3,channels[0]),
            block(channels[0])
        )
        self.downblocks = nn.ModuleList([downblock(channels[i],time_emb_dim,channels[i+1]) 
                                         for i in range(total_channels-1)])
        self.convmid = nn.Sequential(
            block(channels[-1]),
            AttentionResidual(channels[-1],num_heads=16),
            nn.ConvTranspose2d(channels[-1],channels[-2],kernel_size=4,stride=2,padding=1), # double the image size
            nn.BatchNorm2d(channels[-2]),
            nn.ReLU()
        )
        self.upblocks = nn.ModuleList([upblock(channels[i],time_emb_dim,channels[i-1]) 
                                       for i in reversed(range(1,total_channels-1))])
        
        self.convfinal = nn.Sequential(
            block(channels[0]<<1,channels[0]),
            block(channels[0]),
            nn.Conv2d(channels[0],3,1)
        )

    def forward(self,x,time):
        t = self.time_mlp(time)

        residual = []
        x = self.conv0(x)
        for downblock in self.downblocks:
            residual.append(x)
            x = downblock(x,t)
        x = self.convmid(x)
        for upblock in self.upblocks:
            residual_x = residual.pop()
            x = torch.cat((x,residual_x),dim=1) # dimension channel C
            x = upblock(x,t)
        residual_x = residual.pop()
        x = torch.cat((x,residual_x),dim=1)
        return self.convfinal(x)

@torch.no_grad()
def inference(model, x, beta, t=300):
    """
    x: should have the shape of CHW, white noise
    model: pretrained myUnet model
    t: backward timestep
    """
    alpha = 1 - beta
    alpha_cumprod = alpha.cumprod(0)
    var_t = beta[1:] * (1-alpha_cumprod[:-1]) / (1-alpha_cumprod[1:])
    sigma_t = torch.sqrt(var_t)
    param1 = 1./alpha.sqrt()
    param2 = 1./(1-alpha_cumprod).sqrt()
    ISBATCH  = (len(x.shape)==4)
    # for current_time in reversed(range(1,t)):
    for current_time in reversed(range(t)):
        if ISBATCH:
            ct = torch.ones((x.shape[0],),device=x.device) * current_time
        else:
            ct = current_time
        model_noise = model(x,ct)
        mean_prev_t = param1[current_time] * (x-beta[current_time]*param2[current_time]*model_noise)
        x = mean_prev_t + sigma_t[current_time-1] * torch.randn_like(mean_prev_t)

    return mean_prev_t


def train_model(num_epochs, model, loss_func, train_loader, 
                beta, lr=5e-5, device='cpu', verbose=True):
    """This function is modified from file cars.py"""
    model.train()
    optimizer = optim.Adam(model.parameters(),lr=lr)

    alpha = 1 - beta
    cumprod_alpha = alpha.cumprod(0)
    timesteps = beta.shape[0]
    batch_size = train_loader.batch_size

    if verbose:
        import os
        verbosefile = getfilename()+'_'+'verbose.txt'
        if os.path.isfile(verbosefile):
            os.remove(verbosefile)
    
    for epoch in range(num_epochs):
        tic = timer()
        loss_history = []
        for batch_id,(img,_) in enumerate(train_loader):

            # if not batch_id%100:
            #     print(f"{batch_id:4d}\t",datetime.now())

            img = img.to(device=device)
            t = torch.randint(0,timesteps,(batch_size,),device=device)
            img_t,noise_t = image_forward_sampler(img,t,cumprod_alpha)
            noise_model = model(img_t,t)
            loss = loss_func(noise_model,noise_t)
            loss_history.append(loss.data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        toc = timer()
        print(datetime.now())
        if verbose:
            losses = torch.FloatTensor(loss_history)
            verbose_str = f"Train Epoch: {epoch:3d} \tTrain Loss: {losses.mean():.4f} \tTime Elapsed: {toc-tic:.2f}s"
            print(verbose_str)
            with open(verbosefile,'a') as f:
                f.write(verbose_str+'\n')
        
        if not (epoch+1)%50:
            torch.save(model.cpu().state_dict(),getfilename()+'_state_dict_'+str(epoch+1)+'.pt')
            model.to(device)
            print('Model saved!')
        
        if not epoch%25:
            savefigures(model,epoch,device)
            model.train()
            print('Figure saved!')
        
def savefigures(model,epoch,device):
    timesteps = 300
    beta = torch.linspace(.0001, .02, timesteps).to(device=device)
    img_size = 64

    rows = 3
    cols = 4
    num_images = rows*cols
    x = torch.randn((num_images,3,img_size,img_size),device=device)
    img = inference(model,x,beta,300)
    reverse_transform = imgTransform(img_size,True)
    img = img.cpu()
    for i in range(num_images):
        curimg = img[i,...]
        curimg = reverse_transform(curimg)
        plt.subplot(rows,cols,i+1)
        plt.imshow(curimg)
        plt.axis('off')
    plt.savefig('simpleunet_'+str(epoch)+'.png')


@torch.no_grad()
def image_forward_sampler(img,t,cumprod_alpha):
    """
    Inputs: t ranges [0,timesteps)
    """
    at = cumprod_alpha[t] # alphat bar t
    if len(img.shape)==4:
        at = at[...,None,None,None]
    noise = torch.randn_like(img)
    return torch.sqrt(at)*img + torch.sqrt(1-at)*noise, \
            noise

def getfilename(fullname=None):
    if not fullname:
        fullname = __file__
    import os
    basename = os.path.basename(fullname)
    basename = '.'.join(basename.split('.')[:-1])
    return basename

    
if __name__=="__main__":
    batch_size = 128
    img_size = 64
    dtype = torch.float32
    device = "cuda"
    lr = 5e-5

    timesteps = 300
    beta = torch.linspace(.0001, .02, timesteps).to(device=device)
    alpha = 1 - beta
    cumprod_alpha = alpha.cumprod(0)

    # loss_func = nn.MSELoss()
    loss_func = nn.L1Loss()
    train_loader = getdataloader(img_size,batch_size)

    # model = myUnet().to(device=device)
    model = myUnet()
    model.load_state_dict(torch.load('ddpm_state_dict_50.pt'))
    model.to(device)

    print(datetime.now())
    train_model(400,model,loss_func,train_loader,beta,lr,device)
    print(datetime.now())

    # model = myUnet()
    # model.load_state_dict(torch.load('ddpm_state_dict_400.pt'))
    # model.eval()
    # model.to(device)