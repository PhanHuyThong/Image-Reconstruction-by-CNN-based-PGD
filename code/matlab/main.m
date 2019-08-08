function main(configuration_file)
cfg = load_config(configuration_file);

%fn = figure number, used to mark the figures throughout the code
global fn;
fn = 1;

%% components of the imaging operator H and its transpose HT
if cfg.mask %if there is PATH to a mask (a mask is used in the imaging operator)
    global mask;
    mask = load(cfg.mask);
    mask = cell2mat(struct2cell(mask));
end

if cfg.weight
    global weight;
    weight = load(cfg.weight);
    weight = cell2mat(struct2cell(weight));
    weight = permute(weight, [3,4,2,1]);
end
%%%%%%%%%%%%%%%%%%%%%%%%% edit here for extra components of the H, HT
%%%%%%%%%%%%%%%%%%%%%%%%% operators

%% set operator H, HT. Edit here with elseif for new operators
if strcmp(cfg.operator, 'MRI')
    H = @H_MRI;
    Ht = @HT_MRI;
elseif strcmp(cfg.operator, 'Convolution')
    H = @H_CONV;
    Ht = @HT_CONV;
    
end
%% main 
s = System(cfg);

if strcmp(cfg.task, 'test')
    [snr, rec] = s.test(s.x0, s.t, cfg.gamma, H, Ht);
    [snr0, x0, ~] = RSNR(s.x0, s.t);
    figure_maker({x0, rec, s.t}, {num2str(snr0), num2str(snr), 'clean'});

elseif strcmp(cfg.task, 'reconstruct')    
    [snr, rec] = s.reconstruct(s.x0, s.t, cfg.gamma0, H, Ht);
    [snr0, x0, ~] = RSNR(s.x0, s.t);
    figure_maker({x0, rec, s.t}, {num2str(snr0), num2str(snr), 'clean'});    
    savefig([cfg.fig_save_path, 'reconstruct.fig']);
end

    