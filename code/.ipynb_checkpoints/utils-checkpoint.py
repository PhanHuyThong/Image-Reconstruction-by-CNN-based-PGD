# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:59:05 2019

@author: Phan Huy Thong
"""

import torch
from torch import nn
import scipy.stats as stats
import scipy.io
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#==================Unet utils====================
class conv(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding, momentum):
        super(conv, self).__init__()
        self.c = nn.Sequential(nn.Conv2d(inC, outC, kernel_size=kernel_size, padding=padding),
                               nn.BatchNorm2d(outC, momentum=momentum),
                               nn.ReLU())    
    def forward(self, x):
        x = self.c(x)
        return x

class convT(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding, stride, momentum):
        super(convT, self).__init__()
        self.cT = nn.Sequential(nn.ConvTranspose2d(inC, outC, kernel_size=kernel_size, 
                                                   padding=padding,
                                                   stride=stride),
                                nn.BatchNorm2d(outC, momentum=momentum),
                                nn.ReLU())
    def forward(self, x):
        x = self.cT(x)
        return x

class double_conv(nn.Module):
  #conv --> BN --> ReLUx2
    def __init__(self, inC, outC, kernel_size, padding, momentum):
        super(double_conv, self).__init__()
        self.conv2x = nn.Sequential(
            conv(inC, outC, kernel_size=kernel_size, padding=padding, momentum=momentum),
            conv(outC, outC, kernel_size=kernel_size, padding=padding, momentum=momentum))   
    def forward(self, x):
        x = self.conv2x(x)
        return x

class inconv(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding, momentum):
        super(inconv, self).__init__()
        self.conv = double_conv(inC, outC, kernel_size=kernel_size, padding=padding, momentum=momentum)
    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, inC, outC, momentum):
        super(down, self).__init__()
        #go down = maxpool + double conv
        self.go_down = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(inC, outC, kernel_size=3, padding=1, momentum=momentum))   
    def forward(self, x):
        x = self.go_down(x)
        return x

class up(nn.Module):
    def __init__(self, inC, outC, momentum):
        super(up, self).__init__()
        #go up = conv2d to half C-->upsample
        self.convt1 = convT(inC, outC, kernel_size=2, padding=0, stride=2, momentum=momentum)
        self.conv2x = double_conv(inC, outC, kernel_size=3, padding=1, momentum=momentum)
    
    def forward(self, x1, x2):
        #x1 is data from a previous layer, x2 is current input
        x2 = self.convt1(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2x(x)
        return x

class outconv(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding, momentum):
        super(outconv, self).__init__()
        self.conv = conv(inC, outC, kernel_size=kernel_size, padding=padding, momentum=momentum) 
    def forward(self, x):
        x = self.conv(x)
        return x
    
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
#=================Unet=====================
class Unet(nn.Module):
    def __init__(self, inC, outC, momentum=0.1):
        super(Unet, self).__init__()
        self.inc = inconv(inC, 16, 3, 1, momentum)
        self.down1 = down(16, 32, momentum)
        self.down2 = down(32, 64, momentum)
        self.up1 = up(64, 32, momentum)
        self.up2 = up(32, 16, momentum)
        self.outc = outconv(16, outC, 3, 1, momentum)
    def forward(self, x):
        x_input = x.clone().detach()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x2, x3)
        x = self.up2(x1, x)
        x = self.outc(x)
        return x+x_input
#=====================Dataset==========================    
class mydataset(torch.utils.data.Dataset):
    '''
    class mydataset(input_path, target_path, length): dataset[i] returns (input, target) each of size [c,h,w]
    Input: 
        input_path, target_path: type string, the PATH from the folder containing the code being run to the folder 
                                 containing the input/target samples.
        length : type int, the length of the dataset
    
    Example: 
    
    input_path = '../data/train input/'
    target_path = '../data/train target/'
    dataset = mydataset(input_path=input_path, target_path=target_path, length=500)
    dataset[0] --> 1st sample (1.mat) represented as a tuple of (input, target)
    
    Note: in the folder corresponding to the path, the samples are presented as 1.mat, 2.mat, ...
          the file (idx.mat in input_path) is the noisy sample of (idx.mat in target_path)

    '''
    def __init__(self, input_path, target_path, length):
        super(mydataset, self).__init__()
        self.input_path = input_path
        self.target_path = target_path
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        input_loadpath = self.input_path + str(idx+1) +'.mat'
        input_image = torch.tensor(scipy.io.loadmat(input_loadpath)['x'])
        target_loadpath = self.target_path + str(idx+1) + '.mat'
        target_image = torch.tensor(scipy.io.loadmat(target_loadpath)['x'])
        input_image = input_image.float()
        target_image = target_image.float()
        if len(input_image.size())==2:
            input_image.unsqueeze_(0)
        if len(target_image.size())==2:         
            target_image.unsqueeze_(0)
            
        return input_image.to(device), target_image.to(device) #[c,h,w]   
    
#====================Others==============================
def plot_sub_fig(x, row, col, fig_size=None, target_range=True, title=None, save_path=None):
    '''
    Function plot_sub_fig : plot several sub-figures in 1 figure
    
    Input:
        x : type torch.tensor of size [num_sub_figures, h, w], what is plotted. The last element: x[num_sub_figures-1] is the "target", or the reference point
        row, col : type int, number of rows, columns of the subplot
        fig_size : type tuple of int, determines the size of the plot: fig = plt.figure(figsize=fig_size), default is (3*row*col, 3*row*col)
        target_range : type bool, if True: all plots have vmin, vmax =target's min, max         
        title : type typle of strings, the titles of the figures in x
        save_path : PATH including file name to save the figure
        
    Example:
        x1 = torch.rand(1, 5,5)
        x2 = torch.rand(1, 5,5)
        x = torch.cat([x1, x2], dim=0)
        title = ('x1', 'x2')
        plot_sub_fig(x, fig_size=None, target_range=True, title=title, save_path='example.png')
    '''
    with torch.no_grad():
        if fig_size:
            fig = plt.figure(figsize=fig_size)
        else: fig = plt.figure(figsize=(3*row*col, 3*row*col))
    x = x.view(-1,x.size(-2),x.size(-1)).cpu()
    vmin = None
    vmax = None
    if target_range:
        vmin = x[-1].min()
        vmax = x[-1].max()
    for i in range(x.size(0)):
        fig.add_subplot(row, col, i+1)
        plt.grid(False)
        plt.axis('off')
        plt.imshow(x[i], cmap='Greys', vmin=vmin, vmax=vmax)
        if title: 
            plt.title(title[i])
    plt.show()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)

def RSNR(rec,target):
    with torch.no_grad():
        rec = compress3d(rec)
        target = compress3d(target)
        
        rec_size = rec.size()
        rec = rec.cpu().numpy()
        target = target.cpu().numpy()
        rec=rec.reshape(np.size(rec))
        target=target.reshape(np.size(rec))
        slope, intercept, _, _, _ = stats.linregress(rec,target)
        rec=slope*rec+intercept

        return 10*np.log10(sum(target**2)/sum((rec-target)**2)), torch.Tensor(rec).view(rec_size).to(device)
    
def compress2d(x):
    '''compress [1,c,h,w] to [h,w] by root-mean-square at dimension 1, without keeping the dimension, then squeeze dimension 0 '''
    if len(x.size())>2:
        return torch.sqrt(x.pow(2).sum(dim=1))[0]
    return x
def compress3d(x):
    '''compress [1,c,h,w]-->[1,h,w] by root-mean-square at dimension 1, without keeping the dimension'''
    if len(x.size())>3:
        return torch.sqrt(x.pow(2).sum(dim=1))
    return x
def compress4d(x):
    '''compress [1,c,h,w]-->[1,1,h,w] by root-mean-square at dimension 1 and keep the dimension'''
    return torch.sqrt(x.pow(2).sum(dim=1, keepdim=True))


#==================Complex h_mri, ht_mri=====================
def complex_split(x):
    '''x=[n,2 or 1,h,w]--> [n, 1, h, w, 2]'''
    y = torch.empty(x.size(0), 1, x.size(2), x.size(3), 2).to(device)
    y[:,0,:,:,0] = x[:,0,:,:]
    if x.size(1)==1:
        y[:,0,:,:,1] = 0
    else:
        y[:,0,:,:,1] = x[:,1,:,:]
    return y

def complex_merge(x):
    '''x=[n,1,h,w,2] --> [n, 2, h, w]'''
    y = torch.empty(x.size(0), 2, x.size(2), x.size(3)).to(device)
    y[:,0,:,:] = x[:,0,:,:,0]
    y[:,1,:,:] = x[:,0,:,:,1]
    return y

def roll_n(X, axis, n):
    '''x=[n,1,h,w,2]'''
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def cor_cen(x):
    '''x=[n,1,h,w,2]'''
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def cen_cor(x):
    '''x=[n,1,h,w,2]'''
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1) # last dim=2 (real&imag)
def fft(x):
    '''x.size()=[n, c=2, h, w]'''
    #change to [n,1,h,w,2]
    x = complex_split(x)
    #center_to_topleft
    x = cen_cor(x)
    #fft
    x = torch.fft(x, 2, normalized=True)
    #topleft to center
    x = cor_cen(x)
    #merge back to [n, 2, h, w]
    x = complex_merge(x)
    return x.to(device)
def ifft(x):
    '''x.size()=[1, 2, h, w]'''
    #change to [n,1,h,w,2]
    x = complex_split(x)
    #center_to_topleft
    x = cen_cor(x)
    #ifft
    x = torch.ifft(x, 2, normalized=True)
    #topleft to center
    x = cor_cen(x)
    #merge back to [1, 2, h, w]
    x = complex_merge(x)
    return x.to(device)
def h_mri(x, mask):
    '''x[1,1,h,w]-->[1,2,h,w]'''
    x = mask*fft(x)
    return x
def ht_mri(x, mask):
    '''x[1,2,h,w]-->[1,1,h,w]'''
    x = ifft(mask*x)
    return compress4d(x)
