function y = H_CONV(x)
global weight;
y = conv2(x, weight, 'valid');
end
