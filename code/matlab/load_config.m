function r = load_config(filename)
r = ini2struct(filename);

r.preview_data = str2num(r.preview_data);


if strcmp(r.task, 'test')
    r.n_test_samples = str2num(r.n_test_samples);
    r.h = str2num(r.h);
    r.w = str2num(r.w);
    r.test_loss = str2num(r.test_loss);
    r.log_step = str2num(r.log_step);
    r.dk = str2num(r.dk);
    r.dgamma = str2num(r.dgamma);
    r.n_test_iter = str2num(r.n_test_iter);
    r.c = str2num(r.c);
    r.alpha = str2num(r.alpha);
    r.gamma = str2num(r.gamma);
elseif strcmp(r.task, 'reconstruct')
    r.n_test_samples = str2num(r.n_test_samples);
    r.h = str2num(r.h);
    r.w = str2num(r.w);
    r.test_loss = str2num(r.test_loss);
    r.log_step = str2num(r.log_step);
    r.dk = str2num(r.dk);
    r.dgamma = str2num(r.dgamma);
    r.n_test_iter = str2num(r.n_test_iter);
    r.c = str2num(r.c);
    r.alpha = str2num(r.alpha);
    r.gamma0 = str2num(r.gamma0);    
    r.tol = str2num(r.tol);
    r.plot_gamma_snr = str2num(r.plot_gamma_snr);
end


end
