function y = H_MRI(x)
% x real, [w,h,c,n]
% y complex, [w,h,c,n]
global mask;
y = mask.*ifftshift(fft2(fftshift(x)));
end