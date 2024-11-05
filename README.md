# IRGAN

If you want to download checkpoints to run the test download and unzip this folder at the root of the repository: https://drive.google.com/file/d/1boyBtcgvJS1l4zJOONlcMthV7fpZwYJo/view?usp=share_link

We  provide PyTorch implementations for IRGAN. 

#Train

python train.py --dataroot ./datasets/VEDAI --name VEDAI_IRGAN --model IRGAN --direction BtoA

#Test

python test.py --dataroot ./datasets/VEDAI --name VEDAI_IRGAN --model IRGAN --direction BtoA

#Acknowledgments

Our code is inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix , https://github.com/NVIDIA/pix2pixHD and https://github.com/facebookresearch/ConvNeXt.
