# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:45:50 2019

@author: Phan Huy Thong
"""

# import warnings
# warnings.filterwarnings('ignore')

import copy
import torch
from torch import nn
import scipy.stats as stats
import scipy.io
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def prepare_test(model, testloader):
    '''forward-pass the testing samples into the model in training stage to adjust Batch Norm parameters. See    
       https://discuss.pytorch.org/t/model-eval-gives-incorrect-loss-for-model-with-batchnorm-layers/7561/2
       
       Input: model of class Unet
              testloader of class torch.utils.data.Dataloader
       
       Output: model in evaluation mode, with batch norm weights re-adjusted to fit the test samples    
    '''
    model.train()
    for _ in range(20):
        input, target = next(iter(testloader))
        output = model(input)
    model = model.eval()
    
class System():
    '''
        main class, each method is a task to perform, specified in option "task" in config file
    '''
    def __init__(self, cfg):
        self.args = cfg
        self.init_data()
        self.init_para()
        self.init_net()
            
    def init_data(self):
        '''if train
               create: training dataset, dataloader
               create: testing dataset, dataloader for function prepare_test
           if test
               create: only testing dataset, dataloader
           show a pair of input/target if requested. Option not available for task = 'overall snr increase'
        '''
        print('initializing data')
        if self.args.main_task == 'train':
            self.trainset = utils.mydataset(input_path=self.args.train_input_path, 
                                       target_path=self.args.train_target_path, 
                                       length=self.args.n_train_samples)
            self.trainloader = torch.utils.data.DataLoader(self.trainset, 
                                                           batch_size=self.args.train_batch_size,
                                                           shuffle=True)  
            
            self.testset = utils.mydataset(input_path=self.args.test_input_path, 
                                          target_path=self.args.test_target_path, 
                                          length=self.args.n_test_samples)
            self.testloader = torch.utils.data.DataLoader(self.testset, 
                                                          batch_size=1,
                                                          shuffle=True) 
            if self.args.preview_data:
                input, target = next(iter(self.trainloader))
                input = input[0,0:1]
                target = target[0,0:1]
                utils.plot_sub_fig(torch.cat([input, target], dim=0), 1, 2, title=('train input', 'train target'),
                                   save_path=self.args.fig_save_path+'train preview.png') 
                input, target = next(iter(self.testloader))
                input = input[0,0:1]
                target = target[0,0:1]
                utils.plot_sub_fig(torch.cat([input, target], dim=0), 1, 2, title=('test input', 'test target'),
                                   save_path=self.args.fig_save_path+'test preview.png')            
        else:
            self.testset = utils.mydataset(input_path=self.args.test_input_path, 
                                          target_path=self.args.test_target_path, 
                                          length=self.args.n_test_samples)
            self.testloader = torch.utils.data.DataLoader(self.testset, 
                                                          batch_size=1,
                                                          shuffle=True) 
            if self.args.task != 'overall snr increase':
                self.x0, self.t = self.testset[self.args.test_sample_id]
            else:
                self.x0, self.t = self.testset[0]
            if self.args.preview_data:
                utils.plot_sub_fig(torch.cat([self.x0, self.t], dim=0), 1, 2, title=('test input', 'test target'),
                                   save_path=self.args.fig_save_path+'test preview.png')
            if len(self.x0.size()) == 3:
                self.x0.unsqueeze_(dim=0)
                self.t.unsqueeze_(dim=0)
 
        
    def init_para(self):
        '''
        init some parameters needed in the process:
           if in training, create 
               self.dummy_input : a random tensor of the same size as the input
               self.loss_plot   : a list storing the loss values for plotting vs epoch
               self.criterion   : loss function, default is torch.nn.MSELoss()
           if in testing, if there is a mask in the imaging operator, create self.mask
        '''
        
        print('initializing parameters') 
        if self.args.main_task == 'train':
            self.dummy_input = torch.rand(1, self.args.inC, self.args.h, self.args.w).to(device)
            self.loss_plot = []
            if self.args.criterion == 'MSE':
                self.criterion = nn.MSELoss()
            else:
                raise Exception('Criterion not recognized')            
        else:
            if self.args.mask:
                self.mask = torch.tensor(scipy.io.loadmat(self.args.mask)['mask']).float().to(device)
            if self.args.weight:
                self.weight = torch.tensor(scipy.io.loadmat(self.args.weight)['w']).float().to(device)
        
        

                
    def init_net(self):
        '''
        init the CNNs
           create self.net     : the net used in training or testing
                  self.net_tmp : the net used to copy from self.net, then run prepare_test on it to adjust batch norm  
                                 parameters then save in .pth and .onnx format if requested
        self.net is created new or loaded from another .pth file if net_load_path is specified in config
        '''
        print('initializing net')
        if self.args.main_task == 'train':
            if self.args.net == 'Unet':                                             
                self.net = utils.Unet(inC=self.args.inC,
                                       outC=self.args.outC,
                                       momentum=self.args.train_momentum).to(device)

                self.net_tmp = utils.Unet(inC=self.args.inC,                        
                                          outC=self.args.outC,
                                          momentum=self.args.test_momentum).to(device)
                
            ##########################################    add elif here if using another net
            else:
                raise Exception('Neural net not recognized')
        
            if self.args.net_load_path:                                                
                checkpoint = torch.load(self.args.net_load_path)
                self.net.load_state_dict(checkpoint['net_state_dict'])
        else:
            if self.args.net == 'Unet':                                             
                self.net = utils.Unet(inC=self.args.inC,
                                       outC=self.args.outC,
                                       momentum=self.args.test_momentum).to(device) 
            else:
                raise Exception('Neural net not recognized')
                
            checkpoint = torch.load(self.args.net_load_path)
            self.net.load_state_dict(checkpoint['net_state_dict'])
                
    
    def init_optimizer(self, lr):
        print('initializing optimizer')
        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        elif self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        else:
            raise Exception('Optimizer not recognized')

        if self.args.optimizer_load_path:                                           #Option to load a model 
            checkpoint = torch.load(self.args.optimizer_load_path)            
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])      
                   
        
    def train(self, stage):
        '''
        The train function

        Input: 
            stage: type int, value = either 1,2 or 3, specify the training stage. The loss differs in each stage:
                stage 1: loss1 = criterion(output1, target)
                stage 2: loss2 = (criterion(output1, target) + criterion(output2, target))/2
                stage 3: loss = (criterion(output1, target) + criterion(output2, target) + criterion(output3, target))/3

                where output1 = model(inp)
                      output2 = model(output1)
                      output3 = model(target)

        '''
        #init 
        if stage == 1:
            print('input --> target (1)')
            n_epoch = self.args.n_epoch1
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.d_e1, gamma=self.args.d_lr1)
            train_save = self.args.train1_save
            test_save = self.args.test1_save
        elif stage == 2:
            print('stage 2: input --> target, output-->target (2)')
            n_epoch = self.args.n_epoch2
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.d_e2, gamma=self.args.d_lr2)
            train_save = self.args.train2_save
            test_save = self.args.test2_save
        else:
            print('stage 3: input --> target, output-->target, target-->target (3)')
            n_epoch = self.args.n_epoch3
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.d_e3, gamma=self.args.d_lr3)
            train_save = self.args.train3_save
            test_save = self.args.test3_save
        
        #main loop
        for e in range(n_epoch):
            loss_per_epoch = 0.0
            for input, target in self.trainloader:               
                #calculate loss (differs depending on the stage)
                output = self.net(input)
                if stage == 2:
                    input1 = output.clone().detach()
                    output1 = self.net(input1)
                    loss = (self.criterion(output, target) + self.criterion(output1, target))/2    
                elif stage == 3:
                    input1 = output.clone().detach()
                    output1 = self.net(input1)
                    output2 = self.net(target.clone().detach())
                    loss = (self.criterion(output, target)
                            + self.criterion(output1, target) 
                            + self.criterion(output2, target))/3  
                else:
                    loss = self.criterion(output, target)                    
                loss_per_epoch += loss.item()                
                #backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()            
            self.loss_plot.append(loss_per_epoch/len(self.trainloader))
            
            #log
            if self.args.print_loss:
                if e%self.args.log_step == self.args.log_step-1:
                    print('epoch[%d] average loss: %f'% (e+1, self.loss_plot[-1]))
            
            #reduce lr
            scheduler.step()
            
        #save the model
        if train_save:
            torch.save({'net_state_dict': self.net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss_plot': self.loss_plot},
                        self.args.save_path+str(stage)+'-train.pth')
        if test_save:                   
            self.net_tmp.load_state_dict(copy.deepcopy(self.net.state_dict()))
            prepare_test(self.net_tmp, testloader=self.testloader)
            torch.save({'net_state_dict': self.net_tmp.state_dict()}, self.args.save_path+str(stage)+'-test.pth') 
            torch.onnx.export(self.net_tmp, self.dummy_input, self.args.save_path+str(stage)+'.onnx')        
    
    def plot_loss(self):
        '''
        function to plot loss vs epoch
        '''
        fig = plt.figure()
        plt.plot(self.loss_plot)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        if self.args.fig_save_path:
            plt.savefig(self.args.fig_save_path+'train loss.png')
            plt.close(fig)
        
    def test(self, x0, t, gamma, h, ht):
        '''
        Function test: perform RPGD
        Input:
            x0    : type torch.tensor, the noisy image to be reconstructed, normally obtained from HT(measurement)
            t     : type torch.tensor, the target, to calculate RSNR
            gamma : initial learning rate of RPGD
            h     : type function, the H operator defined in main.py
            ht    : type function, the HT operator defined in main.py
            
        Output:
            best_snr : type Float, RSNR of the best reconstruction 
            best_rec : type torch.tensor, the best reconstruction
        '''
        if self.args.task == 'test':
            print('In test:')
            
        c = self.args.c
        alpha = self.args.alpha
        self.net.eval()
        
        with torch.no_grad():
            x = x0.clone().detach()
            y = h(x0)
            best_snr = 0
            for k in range(self.args.n_test_iter):
                z = self.net(x)
                if k>0:
                    if (z-x).pow(2).sum()>c*(z_prev-x_prev).pow(2).sum():
                        alpha = c*(z_prev-x_prev).pow(2).sum()/(z-x).pow(2).sum()*alpha
                x = alpha*z + (1-alpha)*x
                x = x - gamma*ht(h(x)-y)    
                loss = (h(x)-y).pow(2).sum()
                snr, rec = utils.RSNR(x, t)
                if snr>best_snr:
                    best_snr, best_rec = snr, rec  
                if (self.args.test_loss) and (k%10==9):
                    print('iteration:', k+1)
                    print('loss = ', loss.item())
                    print('max pixel value = ', rec.abs().max().item())
                    print('alpha = ', float(alpha))
                    print('snr = ', snr)
                    print('\n')
                z_prev = z
                x_prev = x
                if k%self.args.dk==self.args.dk-1: 
                    gamma /= self.args.dgamma
        return best_snr, best_rec
    
    def reconstruct(self, x0, t, h, ht, idx=''):
        '''
        Function reconstruct: sweep gamma0 provided in config file to find the best gamma
                              then focus on a small interval around it, with 5 mesh points
                              and repeat the sweep until the interval is < tol, where tol is given in cfg file
        Input:
            x0    : type torch.tensor, the noisy image to be reconstructed, normally obtained from HT(measurement)
            t     : type torch.tensor, the target, to calculate RSNR
            h     : type function, the H operator defined in main.py
            ht    : type function, the HT operator defined in main.py
            idx   : type string, to name the saved figures
        Output:
            best_snr : type Float, RSNR of the best reconstruction 
            best_rec : type torch.tensor, the best reconstruction
         '''
        print('reconstructing sample '+idx)
        l = self.args.gamma0[0]
        r = self.args.gamma0[-1]
        snr_plot = np.array([])
        gamma = np.array([])
        gamma_tmp = self.args.gamma0
        while abs(r-l)>self.args.tol:
            best_snr = 0
            gamma = np.append(gamma, gamma_tmp)
            for i in gamma_tmp:
                snr, rec = self.test(x0=x0, t=t, gamma=i, h=h, ht=ht)
                if snr>best_snr: 
                    best_snr, best_rec = snr, rec
                    best_gamma = i
                snr_plot = np.append(snr_plot, snr)
            d = (r-l)/5
            l = best_gamma -d
            r = best_gamma +d
            gamma_tmp = np.linspace(l, r, 4)
        arg = np.argsort(gamma)
        
        if self.args.plot_gamma_snr:
            fig = plt.figure()
            plt.plot(gamma[arg], snr_plot[arg])
            plt.xlabel('gamma')
            plt.ylabel('snr')
            plt.show()
            if self.args.fig_save_path:
                plt.savefig(self.args.fig_save_path+'PGD loss '+idx+'.png')
                plt.close(fig)
            
        return best_snr, best_rec
    
    
    
                    
    
        
            
                