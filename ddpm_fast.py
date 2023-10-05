
import torch
import torch.nn.functional as F
from torchvision import transforms as T, datasets
from torch.utils.data import ConcatDataset
from torch import optim,nn
# import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as timer, sleep
from datetime import datetime
from ddpm import getfilename, image_forward_sampler, savefigures

def fast_savedata(img_size=64):
    transform = T.Compose([
        T.Resize((img_size,img_size)),
        T.PILToTensor()
    ])
    dset1 = datasets.StanfordCars("",transform=transform)
    dset2 = datasets.StanfordCars("",transform=transform,split='test')
    dset = ConcatDataset((dset1,dset2))

    num_images = len(dset)
    img_tensor = torch.empty((num_images,3,img_size,img_size),dtype=torch.uint8,device='cpu')
    label_tensor = torch.empty((num_images,),dtype=torch.uint8,device='cpu')

    for i,(img,label) in enumerate(dset):
        if not i%1e3:
            print(f"{i:5d} saved")
        img_tensor[i,...] = img
        label_tensor[i] = label
    
    torch.save((img_tensor,label_tensor),'Cars_'+'uint8_'+'size'+str(img_size)+'.pt')

def fast_transform():
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.Lambda(lambda x: x.to(dtype=torch.float32).div(255) * 2 - 1)
    ])
    return transform

def fast_reverse_transform():
    transform = T.Compose([
        T.Lambda(lambda x: (x+1)/2), # rescale to [0,1]
        T.Lambda(lambda x: x.permuter(1,2,0).clamp(min=0,max=1)) # CHW to HWC, for imshow
    ])
    return transform

def fast_loaddata():
    return torch.load('Cars_uint8_size64.pt')
    
def fast_train(num_epochs, model, loss_func, img_tensor, 
               beta, batch_size=128, lr=5e-5, device='cpu', verbose=True):
    """This function is modified from file ddpm.py"""
    model.train()
    optimizer = optim.Adam(model.parameters(),lr=lr)

    alpha = 1 - beta
    cumprod_alpha = alpha.cumprod(0)
    timesteps = beta.shape[0]
    
    num_batches = img_tensor.shape[0] // batch_size
    transform = fast_transform()

    if verbose:
        import os
        verbosefile = getfilename(__file__)+'_'+'verbose.txt'
        if os.path.isfile(verbosefile):
            os.remove(verbosefile)
    
    for epoch in range(num_epochs):
        tic = timer()
        loss_history = []

        img_tensor = img_tensor[torch.randperm(img_tensor.shape[0]),...] # shuffle images
        for batch_id in range(num_batches):
            img = img_tensor[batch_id*batch_size:(batch_id+1)*batch_size,...]
            # img = img.to(device=device)
            img = transform(img)
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
        
        if not (epoch+1)%25:
            savefigures(model,epoch+1,device)
            model.train()
            print('Figure saved!')
        
        if not (epoch+1)%50:
            torch.save(model.cpu().state_dict(),getfilename(__file__)+'_state_dict_'+str(epoch+1)+'.pt')
            model.to(device)
            print('Model saved! The program will sleep 5 min.')
            sleep(5*60)
        
    if verbose:
        plt.plot(loss)
        plt.xlabel('Epoch')
        plt.ylabel('MAE Loss')
        plt.show()


if __name__=="__main__":
    batch_size = 128
    img_size = 64
    dtype = torch.float32
    device = "cuda"
    lr = 4e-6

    timesteps = 300
    beta = torch.linspace(.0001, .02, timesteps).to(device=device)
    alpha = 1 - beta
    cumprod_alpha = alpha.cumprod(0)

    loss_func = nn.L1Loss() # MAE is better than MSE
    img_tensor,_ = fast_loaddata()

    from ddpm import myUnet
    model = myUnet()
    model.load_state_dict(torch.load('ddpm_fast_state_dict_100.pt'))
    model.to(device)
    img_tensor = img_tensor.to(device)

    print(datetime.now())
    fast_train(400,model,loss_func,img_tensor,beta,lr=lr,device=device)
    print(datetime.now())
