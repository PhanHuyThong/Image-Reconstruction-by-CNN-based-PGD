function x = HT_CONV(y)
global weight;
x = conv2(y, weight);
end
