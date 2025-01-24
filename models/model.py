import torch
import torch.nn as nn
#import torhvision
from mmcv.runner import load_checkpoint
from models.mit import mit_b4
from models.resnet101 import Resnet
import random
import numpy as np


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


class FHE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.option = args.option
        self.enc_opt = args.enc_opt
        self.pre_trained = args.pre_trained
        self.encoder_option = {'mit':mit_b4(),
                        'conv': Resnet(pre_trained=self.pre_trained),
                        'fused':FusedEncoder(args)}                   
        channels_out = args.out_decoder
        self.encoder = self.encoder_option[self.enc_opt]
        self.decoder_option = {'mit':MitDecoder(args),
                            'conv': ConvDecoder(args),
                            'fused':FusedDecoder(args)}
        self.decoder = self.decoder_option[self.enc_opt]

        self.depth_head = DepthHead(channels_out=channels_out)

        if self.option == "full":
            self.segment_head = SegmentHead(channels_out=channels_out,num_classes=args.num_classes)

        if self.pre_trained and self.enc_opt == 'mit':            
            ckpt_path = args.mit_ckpt_path
            load_checkpoint(self.encoder, ckpt_path, logger=None)


    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        out_d = self.depth_head(x)
        if self.option=="full":
            out_s = self.segment_head(x)
            data = {'pred_d': out_d, 'pred_s':out_s}
        else:
            data = {'pred_d': out_d}

        return data

        
class DepthHead(nn.Module):
    def __init__(self,channels_out) -> None:
        super().__init__()
        self.depth_head = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
    
    def forward(self,x):
        out = self.depth_head(x)

        return out
    

class SegmentHead(nn.Module):
    def __init__(self,channels_out,num_classes) -> None:
        super().__init__()

        self.segment_head = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(channels_out, num_classes, kernel_size=1),
            nn.ReLU()
           # Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
    
    def forward(self,x):
        out = self.segment_head(x)

        return out


########################## FUSED MODEL ############################
class FusedEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pre_trained = args.pre_trained

        self.mit = mit_b4()
        if self.pre_trained:            
            ckpt_path = args.mit_ckpt_path
            load_checkpoint(self.mit, ckpt_path, logger=None)
        self.resnet = Resnet(pre_trained=self.pre_trained)

    def forward(self, x):
        mit = self.mit(x)
        conv = self.resnet(x)
        data = {'mit':mit, 'conv':conv}
        return data


class FusedDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()    
        self.ffb_in = args.in_channel_ffb
        self.ffb_out = args.out_channel_ffb

        self.decoder_in = args.in_channel_decoderb
        self.decoder_out = args.out_channel_decoderb

        self.out_channel = args.out_decoder

        self.ff1 = FeatureFusion(self.ffb_in[0],self.ffb_out[0])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.ff2 = FeatureFusion(self.ffb_in[1],self.ffb_out[1])
        self.ff3 = FeatureFusion(self.ffb_in[2],self.ffb_out[2])
        self.ff4 = FeatureFusion(self.ffb_in[3],self.ffb_out[3])

        self.block1 = DecoderBlock(self.decoder_in[0], self.decoder_out[0])
        self.block2 = DecoderBlock(self.decoder_in[1], self.decoder_out[1])
        self.block3 = DecoderBlock(self.decoder_in[2], self.decoder_out[2])

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.decoder_out[2],
                      out_channels=self.out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU())
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, data):
        mit = data['mit']
        conv = data['conv']
        mit_1, mit_2, mit_3, mit_4 = mit
        conv_1, conv_2, conv_3, conv_4 = conv

        ff1_ = self.ff1(mit_4, conv_4)
        ff1_ = self.upsample(ff1_)
        ff2_ = self.ff2(mit_3,conv_3)
        ff3_ = self.ff3(mit_2,conv_2)
        ff4_ = self.ff4(mit_1, conv_1)

        x = self.block1(ff1_,ff2_)
        x = self.block2(x,ff3_)
        x = self.block3(x,ff4_)
        x = self.conv1(x)
        x = self.upsample(x)

        return x


class FeatureFusion(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=int(in_channel/2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel/2)),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel/2),
                      out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU())

    def forward(self, x_mit, x_cnn):
        x_cnn = self.conv1(x_cnn)
        x_cnn = self.conv2(x_cnn)
        x = torch.cat((x_mit,x_cnn),1)
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2,
                      out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU())
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self,ffi_1,ffi_2):
        ffi_1 = self.conv1(ffi_1)
        x = torch.cat((ffi_1,ffi_2),1)
        x = self.conv2(x)
        x = self.upsample(x)

        return x
    

class Fused_Conv(nn.Module):
    def __init__(self, in_channel ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel),
                      out_channels=in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, 
                      out_channels=int(in_channel), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel)),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, 
                      out_channels=int(in_channel), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel)))

        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.sigmoid(x)

        return x
    

########################## Conv MODEL ############################
class ConvDecoderBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel),
                      out_channels=in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, 
                      out_channels=int(in_channel/2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel/2)),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel/2), 
                      out_channels=int(in_channel/2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel/2)))

        self.relu=nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(x)

        return x


class ConvDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ffb_in = args.in_channel_ffb
        self.decoder_in = args.in_channel_decoderb
        self.decoder_out = args.out_channel_decoderb

        self.ff1 = ConvDecoderBlock(self.ffb_in[0])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.ff2 = ConvDecoderBlock(self.ffb_in[1])
        self.ff3 = ConvDecoderBlock(self.ffb_in[2])
        self.ff4 = ConvDecoderBlock(self.ffb_in[3])

        self.block1 = DecoderBlock(self.decoder_in[0], self.decoder_out[0])
        self.block2 = DecoderBlock(self.decoder_in[1], self.decoder_out[1])
        self.block3 = DecoderBlock(self.decoder_in[2], self.decoder_out[2])

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)



    def forward(self, x):
        x_1, x_2, x_3, x_4 = x
        ff1_ = self.ff1(x_4)
        ff1_ = self.upsample(ff1_)
        ff2_ = self.ff2(x_3)
        ff3_ = self.ff3(x_2)
        ff4_ = self.ff4(x_1)

        x = self.block1(ff1_,ff2_)
        x = self.block2(x,ff3_)
        x = self.block3(x,ff4_)
        x = self.conv1(x)
        x = self.upsample(x)

        return x
    

################################# MiT Model ##########################################
class MitDecoderBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel),
                      out_channels=in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channel, 
                      out_channels=int(in_channel*2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel*2)),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel*2), 
                      out_channels=int(in_channel*2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel*2)))

        self.relu=nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(x)

        return x


class MitDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ffb_out = args.out_channel_ffb
        self.decoder_in = args.in_channel_decoderb
        self.decoder_out = args.out_channel_decoderb

        self.ff1 = MitDecoderBlock(self.ffb_out[0])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.ff2 = MitDecoderBlock(self.ffb_out[1])
        self.ff3 = MitDecoderBlock(self.ffb_out[2])
        self.ff4 = MitDecoderBlock(self.ffb_out[3])

        self.block1 = DecoderBlock(self.decoder_in[0], self.decoder_out[0])
        self.block2 = DecoderBlock(self.decoder_in[1], self.decoder_out[1])
        self.block3 = DecoderBlock(self.decoder_in[2], self.decoder_out[2])

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)



    def forward(self, x):
        x_1, x_2, x_3, x_4 = x
        ff1_ = self.ff1(x_4)
        ff1_ = self.upsample(ff1_)
        ff2_ = self.ff2(x_3)
        ff3_ = self.ff3(x_2)
        ff4_ = self.ff4(x_1)

        x = self.block1(ff1_,ff2_)
        x = self.block2(x,ff3_)
        x = self.block3(x,ff4_)
        x = self.conv1(x)
        x = self.upsample(x)

        return x