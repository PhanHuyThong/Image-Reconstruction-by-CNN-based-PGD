function [SNR, x, c] = RSNR(x,t)

if size(x,3)>1
    x = sqrt(sum(x.^2,3));
    t = sqrt(sum(t.^2,3));
end
sumP    =        sum(t(:));
sumI    =        sum(x(:));
sumIP   =        sum( t(:) .* x(:) );
sumI2   =        sum(x(:).^2)           ;
A       =        [sumI2, sumI; sumI, numel(t)];
b       =        [sumIP;sumP]             ;
c       =        (A)\b                    ;
x       =        c(1)*x+c(2)            ;
err     =        sum((t(:)-x(:)).^2)      ;
SNR     =        10*log10(sum(t(:).^2)/err) ;
end