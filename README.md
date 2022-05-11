# Lidar_intensity_modelling
Using GAN to model Lidar intensity based on semantic segmentation layout and 3D spatial coordinates.
## Architecture
![a](Images/training-diagram.png)
## Qualitative Results
![a](Images/Kitti-exp-figure.png)
## Requirements 
torch==1.10.2
torchvision==0.4.0
PyYAML==5.1
`pip install argparse`
## train
`pyhon train.py --cfg_train configs/train_pix2pix.yaml`
## test
`pyhon test.py --cfg_train configs/test_pix2pix.yaml`
