import configparser

#converters
def string2tuple(x):
    return tuple(map(int, x.split()))

def string2list(x):
    return list(map(float, x.split()))
    
class Config:
    @classmethod
    def load_config(cls, filename):
        config = configparser.ConfigParser(converters={'tuple': string2tuple, 'list': string2list})
        config.read(filename)
        cls.config = config
        
        s = 'general'
        cls.name = config.get(s, 'name')
        cls.main_task = config.get(s, 'main_task')
        cls.task = config.get(s, 'task')
        cls.preview_data = config.getboolean(s, 'preview_data')
        cls.fig_save_path = config.get(s, 'fig_save_path')
        
        if cls.main_task == 'train':
            if cls.task == 'train projector':
                
                s = 'train projector'
                
                cls.net_load_path = config.get(s, 'net_load_path')
                cls.optimizer_load_path = config.get(s, 'optimizer_load_path')
        
                cls.save_path = config.get(s, 'save_path')
                cls.train1_save = config.getboolean(s, 'train1_save')
                cls.train2_save = config.getboolean(s, 'train2_save')
                cls.train3_save = config.getboolean(s, 'train3_save')
                cls.test1_save = config.getboolean(s, 'test1_save')
                cls.test2_save = config.getboolean(s, 'test2_save')
                cls.test3_save = config.getboolean(s, 'test3_save')
        
                cls.train_momentum = config.getfloat(s, 'train_momentum')
                cls.test_momentum = config.getfloat(s, 'test_momentum')
                cls.inC = config.getint(s, 'inC')
                cls.outC = config.getint(s, 'outC')
                cls.net = config.get(s, 'net')

                cls.optimizer = config.get(s, 'optimizer')
                cls.reset_optimizer = config.getboolean(s, 'reset_optimizer')

                cls.criterion = config.get(s, 'criterion')

                cls.plot_loss = config.getboolean(s, 'plot_loss')
                cls.print_loss = config.getboolean(s, 'print_loss')
                cls.log_step = config.getint(s, 'log_step')

                cls.train_input_path = config.get(s, 'train_input_path')
                cls.train_target_path = config.get(s, 'train_target_path')
                cls.test_input_path = config.get(s, 'test_input_path')
                cls.test_target_path = config.get(s, 'test_target_path')        
                cls.n_train_samples = config.getint(s, 'n_train_samples')
                cls.n_test_samples = config.getint(s, 'n_test_samples')
                cls.train_batch_size = config.getint(s, 'train_batch_size')
                cls.h = config.getint(s, 'h')
                cls.w = config.getint(s, 'w')

                cls.n_epoch1 = config.getint(s, 'n_epoch1')
                cls.n_epoch2 = config.getint(s, 'n_epoch2')
                cls.n_epoch3 = config.getint(s, 'n_epoch3')
                cls.lr1 = config.getfloat(s, 'lr1')
                cls.lr2 = config.getfloat(s, 'lr2')
                cls.lr3 = config.getfloat(s, 'lr3')    
                cls.d_e1 = config.getint(s, 'd_e1')
                cls.d_e2 = config.getint(s, 'd_e2')
                cls.d_e3 = config.getint(s, 'd_e3')
                cls.d_lr1 = config.getint(s, 'd_lr1')
                cls.d_lr2 = config.getint(s, 'd_lr2')
                cls.d_lr3 = config.getint(s, 'd_lr3')
                
            elif cls.task == 'train1':
                s = 'train1'
                cls.net_load_path = config.get(s, 'net_load_path')
                cls.optimizer_load_path = config.get(s, 'optimizer_load_path')
        
                cls.save_path = config.get(s, 'save_path')
                cls.train1_save = config.getboolean(s, 'train1_save')
                cls.test1_save = config.getboolean(s, 'test1_save')
        
                cls.train_momentum = config.getfloat(s, 'train_momentum')
                cls.test_momentum = config.getfloat(s, 'test_momentum')
                cls.inC = config.getint(s, 'inC')
                cls.outC = config.getint(s, 'outC')
                cls.net = config.get(s, 'net')

                cls.optimizer = config.get(s, 'optimizer')
                cls.reset_optimizer = config.getboolean(s, 'reset_optimizer')

                cls.criterion = config.get(s, 'criterion')

                cls.plot_loss = config.getboolean(s, 'plot_loss')
                cls.print_loss = config.getboolean(s, 'print_loss')
                cls.log_step = config.getint(s, 'log_step')

                cls.train_input_path = config.get(s, 'train_input_path')
                cls.train_target_path = config.get(s, 'train_target_path')
                cls.test_input_path = config.get(s, 'test_input_path')
                cls.test_target_path = config.get(s, 'test_target_path')        
                cls.n_train_samples = config.getint(s, 'n_train_samples')
                cls.n_test_samples = config.getint(s, 'n_test_samples')
                cls.train_batch_size = config.getint(s, 'train_batch_size')
                cls.h = config.getint(s, 'h')
                cls.w = config.getint(s, 'w')

                cls.n_epoch1 = config.getint(s, 'n_epoch1')
                cls.lr1 = config.getfloat(s, 'lr1') 
                cls.d_e1 = config.getint(s, 'd_e1')
                cls.d_lr1 = config.getint(s, 'd_lr1')
                
            elif cls.task == 'train2':
                s = 'train2'
                cls.net_load_path = config.get(s, 'net_load_path')
                cls.optimizer_load_path = config.get(s, 'optimizer_load_path')
        
                cls.save_path = config.get(s, 'save_path')
                cls.train2_save = config.getboolean(s, 'train2_save')
                cls.test2_save = config.getboolean(s, 'test2_save')
        
                cls.train_momentum = config.getfloat(s, 'train_momentum')
                cls.test_momentum = config.getfloat(s, 'test_momentum')
                cls.inC = config.getint(s, 'inC')
                cls.outC = config.getint(s, 'outC')
                cls.net = config.get(s, 'net')

                cls.optimizer = config.get(s, 'optimizer')
                cls.reset_optimizer = config.getboolean(s, 'reset_optimizer')

                cls.criterion = config.get(s, 'criterion')

                cls.plot_loss = config.getboolean(s, 'plot_loss')
                cls.print_loss = config.getboolean(s, 'print_loss')
                cls.log_step = config.getint(s, 'log_step')

                cls.train_input_path = config.get(s, 'train_input_path')
                cls.train_target_path = config.get(s, 'train_target_path')
                cls.test_input_path = config.get(s, 'test_input_path')
                cls.test_target_path = config.get(s, 'test_target_path')        
                cls.n_train_samples = config.getint(s, 'n_train_samples')
                cls.n_test_samples = config.getint(s, 'n_test_samples')
                cls.train_batch_size = config.getint(s, 'train_batch_size')
                cls.h = config.getint(s, 'h')
                cls.w = config.getint(s, 'w')

                cls.n_epoch2 = config.getint(s, 'n_epoch2')
                cls.lr2 = config.getfloat(s, 'lr2')
                cls.d_e2 = config.getint(s, 'd_e2')
                cls.d_lr2 = config.getint(s, 'd_lr2')
                
            elif cls.task == 'train3':
                s = 'train3'
                cls.net_load_path = config.get(s, 'net_load_path')
                cls.optimizer_load_path = config.get(s, 'optimizer_load_path')
        
                cls.save_path = config.get(s, 'save_path')
                cls.train3_save = config.getboolean(s, 'train3_save')
                cls.test3_save = config.getboolean(s, 'test3_save')
        
                cls.train_momentum = config.getfloat(s, 'train_momentum')
                cls.test_momentum = config.getfloat(s, 'test_momentum')
                cls.inC = config.getint(s, 'inC')
                cls.outC = config.getint(s, 'outC')
                cls.net = config.get(s, 'net')

                cls.optimizer = config.get(s, 'optimizer')
                cls.reset_optimizer = config.getboolean(s, 'reset_optimizer')

                cls.criterion = config.get(s, 'criterion')

                cls.plot_loss = config.getboolean(s, 'plot_loss')
                cls.print_loss = config.getboolean(s, 'print_loss')
                cls.log_step = config.getint(s, 'log_step')

                cls.train_input_path = config.get(s, 'train_input_path')
                cls.train_target_path = config.get(s, 'train_target_path')
                cls.test_input_path = config.get(s, 'test_input_path')
                cls.test_target_path = config.get(s, 'test_target_path')        
                cls.n_train_samples = config.getint(s, 'n_train_samples')
                cls.n_test_samples = config.getint(s, 'n_test_samples')
                cls.train_batch_size = config.getint(s, 'train_batch_size')
                cls.h = config.getint(s, 'h')
                cls.w = config.getint(s, 'w')

                cls.n_epoch3 = config.getint(s, 'n_epoch3')
                cls.lr3 = config.getfloat(s, 'lr3')
                cls.d_e3 = config.getint(s, 'd_e3')
                cls.d_lr3 = config.getint(s, 'd_lr3')                
        else:
            if cls.task == 'test':
                s = 'test'
                cls.net_load_path = config.get(s, 'net_load_path')
                cls.test_momentum = config.getfloat(s, 'test_momentum')
                cls.inC = config.getint(s, 'inC')
                cls.outC = config.getint(s, 'outC')
                cls.net = config.get(s, 'net') 
                
                cls.test_input_path = config.get(s, 'test_input_path')
                cls.test_target_path = config.get(s, 'test_target_path')        
                cls.n_test_samples = config.getint(s, 'n_test_samples')
                cls.h = config.getint(s, 'h')
                cls.w = config.getint(s, 'w')
                cls.test_sample_id = config.getint(s, 'test_sample_id')
                
                cls.test_loss = config.getboolean(s, 'test_loss')
                
                cls.dk = config.getint(s, 'dk')
                cls.dgamma = config.getfloat(s, 'dgamma')
                cls.n_test_iter = config.getint(s, 'n_test_iter')
                cls.c = config.getfloat(s, 'c')
                cls.alpha = config.getfloat(s, 'alpha')
                cls.gamma = config.getfloat(s, 'gamma')

                #add options here for components of the operators (eg. mask for MRI operator, kernel(weight) for 
                #convolutional operator,...)                
                cls.mask = config.get(s, 'mask')
                cls.weight = config.get(s, 'weight')
                cls.operator = config.get(s, 'operator')
                
            elif cls.task == 'reconstruct':
                s = 'reconstruct'
                cls.net_load_path = config.get(s, 'net_load_path')
                cls.test_momentum = config.getfloat(s, 'test_momentum')
                cls.inC = config.getint(s, 'inC')
                cls.outC = config.getint(s, 'outC')
                cls.net = config.get(s, 'net') 
                
                cls.test_input_path = config.get(s, 'test_input_path')
                cls.test_target_path = config.get(s, 'test_target_path')        
                cls.n_test_samples = config.getint(s, 'n_test_samples')
                cls.h = config.getint(s, 'h')
                cls.w = config.getint(s, 'w')
                cls.test_sample_id = config.getint(s, 'test_sample_id')
                
                cls.test_loss = config.getboolean(s, 'test_loss')
                
                cls.dk = config.getint(s, 'dk')
                cls.dgamma = config.getfloat(s, 'dgamma')
                cls.n_test_iter = config.getint(s, 'n_test_iter')
                cls.c = config.getfloat(s, 'c')
                cls.alpha = config.getfloat(s, 'alpha')
                cls.gamma0 = config.getlist(s, 'gamma0')
                cls.plot_gamma_snr = config.getboolean(s, 'plot_gamma_snr')
                cls.tol = config.getfloat(s, 'tol')
                
                #add options here for components of the operators (eg. mask for MRI operator, kernel(weight) for 
                #convolutional operator,...)                
                cls.mask = config.get(s, 'mask')
                cls.weight = config.get(s, 'weight')
                cls.operator = config.get(s, 'operator')  
                
                
            elif cls.task == 'overall snr increase':
                s = 'overall snr increase'
                cls.net_load_path = config.get(s, 'net_load_path')
                cls.test_momentum = config.getfloat(s, 'test_momentum')
                cls.inC = config.getint(s, 'inC')
                cls.outC = config.getint(s, 'outC')
                cls.net = config.get(s, 'net') 
                
                cls.test_input_path = config.get(s, 'test_input_path')
                cls.test_target_path = config.get(s, 'test_target_path')        
                cls.n_test_samples = config.getint(s, 'n_test_samples')
                cls.h = config.getint(s, 'h')
                cls.w = config.getint(s, 'w')
                
                cls.test_loss = config.getboolean(s, 'test_loss')
                
                cls.dk = config.getint(s, 'dk')
                cls.dgamma = config.getfloat(s, 'dgamma')
                cls.n_test_iter = config.getint(s, 'n_test_iter')
                cls.c = config.getfloat(s, 'c')
                cls.alpha = config.getfloat(s, 'alpha')
                cls.gamma0 = config.getlist(s, 'gamma0')
                cls.plot_gamma_snr = config.getboolean(s, 'plot_gamma_snr')
                cls.show_reconstruction = config.getboolean(s, 'show_reconstruction')
                cls.tol = config.getfloat(s, 'tol')
                
                #add options here for components of the operators (eg. mask for MRI operator, kernel(weight) for 
                #convolutional operator,...)
                cls.mask = config.get(s, 'mask')
                cls.weight = config.get(s, 'weight')
                cls.operator = config.get(s, 'operator')               
                