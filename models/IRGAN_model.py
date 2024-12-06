import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from preprocess import PreProcessModel
from . import networks 
import argparse
from torch import nn
from print_if_full import for_preprocessed_image


class IRGANModel(BaseModel):
    def name(self):
        return 'IRGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='uconvnext')
        parser.set_defaults(dataset_mode='aligned')
        if is_train:
            
            parser.set_defaults(pool_size=0, no_lsgan=True)
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_sobel', type=float, default=15.0, help='weight for sobel loss')
            parser.add_argument('--num_D', type=int, default=3, help='scale')
            parser.add_argument('--vnums', type=int, default=4)
            parser.add_argument('--smp_model', type=str, default='Unet')
            parser.add_argument('--smp_encoder', type=str, default='resnet18')
            parser.add_argument('--smp_encoder_weights', type=str, default='imagenet')
        parser.add_argument('--lambda_tevnet', type=float, default=15.0, help='weight for TeVNet loss')
        parser.add_argument('--tevnet_weights', type=str, default=None,  help='path to pretrained TeVNet weights')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        print(opt.preprocess, bool(opt.preprocess))
        self.preprocess = bool(opt.preprocess)
        self.preprocess_type = opt.preprocess
        self.tevnet = opt.tevnet_weights

        print("Tevnet pth = ", self.tevnet)
        self.preprocessed = None
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'G_Sobel', 'G_TeVNet', 'D_real', 'D_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.preprocess:
            self.visual_names.insert(1, 'preprocessed')
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        if self.preprocess:
            self.model_names.append('preprocess')
        # load/define networks


        if self.tevnet:
            from TeVNet.models import TeVNet
            from TeVNet.utils import TeVloss

            self.tevnet_model = TeVNet(in_channels=3, out_channels=2 + opt.vnums, args=opt).to(self.device)
            self.tevnet_model = nn.DataParallel(self.tevnet_model, device_ids=[0])

            state_dict = self.tevnet_model.state_dict()
            for n, p in torch.load(self.tevnet, map_location=lambda storage, loc: storage, weights_only = True)['state_dict'].items():
                if n in state_dict:
                    state_dict[n].copy_(p)
                else:
                    raise KeyError(f"Key '{n}' not found in model state_dict.")

            self.tevnet_model.to(self.device)
            self.tevnet_model.eval()
            self.lossmodule = TeVloss(vnums=opt.vnums)

            print("TeVNet Loaded", flush = True)

        
        if opt.preprocess == 'full':
            opt.input_nc = 16
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            netD_input_nc = opt.input_nc + opt.output_nc

            self.netD = networks.define_D(input_nc=netD_input_nc, ndf=opt.ngf, n_layers_D=opt.n_layers_D, num_D=opt.num_D).to(self.device)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.sobelloss = networks.SobelLoss()


            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
        print(self.preprocess, type(self.preprocess))
        if self.preprocess:
            if opt.preprocess == 'full':
                final_conv = False
            else:
                final_conv = True
            self.netpreprocess = PreProcessModel(3,16, final_conv).to(self.device)
            if self.isTrain:
                self.preprocess_optimizer = torch.optim.Adam(self.netpreprocess.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.preprocess_optimizer)
                self.criterion_preprocess = torch.nn.SmoothL1Loss()




    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        if self.preprocess:
            preprocessed = self.netpreprocess(self.real_A)
            if self.preprocess_type == 'full':
                pass
            else:
                self.preprocessed = preprocessed
            self.fake_B = self.netG(preprocessed)
        else:
            self.fake_B = self.netG(self.real_A)

    ###########
    def test_new(self):
        with torch.no_grad():
            self.forward()
        mse = torch.nn.MSELoss()
        loss_mse = mse(self.real_B, self.fake_B)
        return loss_mse



    def backward_D(self):
        #patchgan
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1)).detach()
        pred_fake = self.netD(fake_AB)
        self.loss_D_fake = self.criterionGAN(pred_fake, False) / self.opt.num_D

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True) / self.opt.num_D

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        #patchgan
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) / self.opt.num_D

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G_Sobel = self.sobelloss(self.fake_B, self.real_B) * self.opt.lambda_sobel

        if self.tevnet:
            with torch.no_grad():
                real_tevnet_output = self.tevnet_model(self.real_B)
                fake_tevnet_output = self.tevnet_model(self.fake_B)
            
            # Compare TeVNet decompositions
            self.loss_G_TeVNet = (
                self.criterionL1(fake_tevnet_output[:, 0:2], real_tevnet_output[:, 0:2]) * 0.5 +  # T and e components
                self.criterionL1(fake_tevnet_output[:, 2:], real_tevnet_output[:, 2:]) * 0.5 +    # V components
                self.lossmodule.loss_rec(fake_tevnet_output, self.fake_B) # Reconstuction loss
            ) * self.opt.lambda_tevnet

        else:
            self.loss_G_TeVNet = 0.0

        self.loss_G = self.loss_G_L1 + self.loss_G_GAN + self.loss_G_Sobel + self.loss_G_TeVNet

        self.loss_G.backward()


    
    def backward_preprocess(self):
        #fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.loss_preprocess = self.criterion_preprocess(self.preprocessed, self.real_A) * self.opt.lambda_L1
        self.loss_preprocess.backward()


    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netG, True)
        # if self.preprocess:
        #     self.set_requires_grad(self.netpreprocess, False)

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        #update preprocess
        if self.preprocess:
            self.preprocess_optimizer.step()



