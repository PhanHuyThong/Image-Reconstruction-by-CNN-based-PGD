# Project Title

Deep-learning-based Projected Gradient Descent for Image Reconstruction 

This project includes a framework to:

1) Train a neural network (a Unet) as an image-to-image projector in Pytorch, export it in .pth and .onnx format
2) Apply the Relaxed Projected Gradient Descent (RPGD) in [1] for image reconstruction. For this part, the code is provided in both Python and Matlab. In Matlab, the measurement operator maybe more readily available thanks to many libraries.


![](header.png) %%% 

## Getting Started

### Prerequisites

Python 3.7
Pytorch 1.1.0
Scipy 1.2.1
Matplotlib 3.0.3

for Matlab code:

Matlab R2019a
Deep Learning Toolbox

### Installing

Download the folders code and data

## Running the tests

In Linux command lines or Windows Shell, go inside the folder code, then type:

```
python main.py ../data/reconstruct.cfg
```

to run a test reconstruction in Python.

To do the same thing in Matlab, go into the folder Matlab, then type:

```
matlab -nosplash -nodesktop -r "main('reconstruct.cfg')"
```


The test run RPGD with a pre-trained net named 3-test.pth if using Python or 3.onnx if using Matlab. The end resuts show 3 images, from left to right: the noisy sample, the reconstructed image, and the clean image, respectively, with the value of RSNR with respect to the clean image displayed on top of the 1st 2 images. 

For more info on the function used, open a Python console or notebook and type:

```
import system
help(system.System.reconstruct)
```

!!!!TODO: help in Matlab

## Detailed Usage Instruction
### Training CNN

#### Data

A customized class named *mydataset* is provided in utils.py to specifically read .mat data. For more info on the class, please type in Python console:

```
import utils
help(utils.mydataset)
```

*Note: The code (for now), only works with data in *mydataset* class. 

#### CNN

The components to build a Unet and a default Unet with 2 times going down is provided in utils.py. To use another net, one can create it in the same module (utils.py), then go to line 148 of system.py to add the option to use that net, and adjust the parameter "net" in the config files correspondingly. 

It should be noted that if the Matlab code is to be used, one should ensure that Pytorch can export such a net to .onnx format, and Matlab can import that .onnx file.

#### Training options

There are 4 options provided, corresponding to 3 config files provided as template:

    train1.cfg          : train the CNN with loss = criterion(output1, target)
    train2.cfg          : train the CNN with loss = (criterion(output1, target) + criterion(output2, target))/2
    train3.cfg          : train the CNN with loss = (criterion(output1, target) + 
                                                     criterion(output2, target) + 
                                                     criterion(output3, target))/3
                  where output1 = model(inp)
                        output2 = model(output1)
                        output3 = model(target)


                                            
    train_projector.cfg : train a projector by going through all 3 options above sequentially.
    
*Note: train1, train2, train3 are normally used to continue the training from a saved CNN; train_projector is like a "convenient package" combining them 

### Image reconstruction

#### In Python 

There are 3 options provided, corresponding to 3 config files provided as template:

    test.cfg        : RPGD for 1 sample at initial learning rate = gamma
    reconstruct.cfg : RPGD for 1 sample with a sweep over initial learning rate = gamma0 , type list
    overall_snr_increase.cfg : go through all the test samples and reconstruct them similar to reconstruct.cfg, print out their average RSNR increase 
    
*Note: the RPGD algorithm requires an operator H representing the measurement process, from which one can obtain HT (H transpose). The code provided H_MRI(x) = mask*fft(x) and HT_MRI(y) = ifft(mask*y); H_conv(x) = torch.nn.functional.conv2d(x, weight) and HT_conv = torch.nn.functional.conv_transpose2d(x, weight). To add another operator, make changes in line 38 main.py and the config file correspondingly.


## Contributing


## Versioning
 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments


ini2struct.m (link: https://ch.mathworks.com/matlabcentral/fileexchange/17177-ini2struct)


* Hat tip to anyone whose code was used
* Inspiration
* etc

