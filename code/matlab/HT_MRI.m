function x = HT_MRI(y)
%x real
global mask;
x = ifftshift(ifft2(fftshift(mask.*y)));
x = real(x);
end
