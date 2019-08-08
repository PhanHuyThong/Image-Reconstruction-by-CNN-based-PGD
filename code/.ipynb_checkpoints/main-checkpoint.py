# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:12:12 2019

@author: Phan Huy Thong
"""

import os, sys
sys.path.append(os.getcwd())
import argparse 
from system import System
from config import Config as cfg
import utils
import torch

if __name__ == '__main__':
    #read argument
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='default.cfg', help='Specify config file', metavar='FILE')
    args = parser.parse_args()
    
    cfg.load_config(args.config)
    
    #run
    s = System(cfg)
    
    if cfg.main_task == 'test':
        if cfg.operator == 'MRI':
            def h(x):
                return utils.h_mri(x, mask=s.mask)
            def ht(x):
                return utils.ht_mri(x, mask=s.mask)
        elif cfg.operator == 'Convolution':
            def h(x):
                return torch.nn.functional.conv2d(x, s.weight)
            def ht(y):
                return torch.nn.functional.conv_transpose2d(y, s.weight)
            #################################################                    ADD elif HERE FOR NEW OPERATORS
        else:
            raise Exception('imaging operator not recognized')

    if cfg.task == 'train projector':
        s.init_optimizer(cfg.lr1)        
        s.train(1)        
        if cfg.reset_optimizer:
            s.init_optimizer(cfg.lr2)
        s.train(2)        
        if cfg.reset_optimizer:
            s.init_optimizer(cfg.lr3) 
        s.train(3)        
        if cfg.plot_loss:
            s.plot_loss()
            
    elif cfg.task == 'train1':
        s.init_optimizer(cfg.lr1)        
        s.train(1)
        if cfg.plot_loss:
            s.plot_loss()        
    elif cfg.task == 'train2':
        s.init_optimizer(cfg.lr2)        
        s.train(2)
        if cfg.plot_loss:
            s.plot_loss()        
    elif cfg.task == 'train3':
        s.init_optimizer(cfg.lr3)        
        s.train(3)
        if cfg.plot_loss:
            s.plot_loss()
    
    elif cfg.task == 'test':
        print('reconstructing at gamma = ', cfg.gamma)
        snr0, x0 = utils.RSNR(s.x0, s.t)
        best_snr, best_rec = s.test(s.x0, s.t, cfg.gamma, h=h, ht=ht)
        utils.plot_sub_fig(torch.cat([utils.compress3d(x0), best_rec, utils.compress3d(s.t)], dim=0), 
                           1,3, title=('RSNR = '+str(round(snr0,2)), 'RSNR = '+str(round(best_snr,2)), 'clean'),
                           save_path=cfg.fig_save_path+'test.png')  
    
    elif cfg.task == 'reconstruct':
        print('sweep gamma to find best reconstruction')
        snr0, x0 = utils.RSNR(s.x0, s.t)
        best_snr, best_rec = s.reconstruct(s.x0, s.t, h=h, ht=ht, idx=str(cfg.test_sample_id+1))
        utils.plot_sub_fig(torch.cat([utils.compress3d(x0), best_rec, utils.compress3d(s.t)], dim=0), 
                       1,3, title=('RSNR = '+str(round(snr0,2)), 'RSNR = '+str(round(best_snr,2)), 'clean'),
                           save_path=cfg.fig_save_path+'reconstruct sample '+str(cfg.test_sample_id+1)+'.png')
      
    elif cfg.task == 'overall snr increase':
        t = 0
        idx = 1
        for s.x0, s.t in s.testloader:
            snr0, x0 = utils.RSNR(s.x0, s.t)
            snr, rec = s.reconstruct(s.x0, s.t, h=h, ht=ht, idx=str(idx))
            if cfg.show_reconstruction:
                utils.plot_sub_fig(torch.cat([utils.compress3d(x0), rec, utils.compress3d(s.t)], dim=0), 
                                   1,3, title=('RSNR = '+str(round(snr0,2)), 'RSNR = '+str(round(snr,2)), 'clean'),
                                   save_path=cfg.fig_save_path+'reconstruct sample '+str(idx)+'.png')
            idx += 1
            t += snr-snr0
        print('average snr increase among all test samples = ', t/cfg.n_test_samples)
    else:
        raise Exception('task not recognized')
        
    
        
    