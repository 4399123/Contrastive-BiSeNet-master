#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
from .caformer_m36 import Caformer_M36
from timm.layers.squeeze_excite import EffectiveSEModule
from torch.nn import BatchNorm2d
from torch.cuda.amp import autocast

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.Hardswish(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.Upsample(scale_factor=up_factor,
                mode='bilinear', align_corners=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x=x.float()
        x = self.up(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1,act=False)
        self.se=EffectiveSEModule(out_chan)
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        out=self.se(feat)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Caformer_M36()
        self.arm16 = AttentionRefinementModule(384, 256)
        self.arm32 = AttentionRefinementModule(576, 256)
        self.conv_head32 = ConvBNReLU(256, 256, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(256, 256, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(576, 256, ks=1, stride=1, padding=0)
        self.up32 = nn.Upsample(scale_factor=2.)
        self.up16 = nn.Upsample(scale_factor=2.)

        # self.init_weight()

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)

        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_sum =feat32_sum.float()  ###
        feat32_up = self.up32(feat32_sum)
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_sum=feat16_sum.float()  ###
        feat16_up = self.up16(feat16_sum)
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up # x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 128, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(128, 128, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(128, 128, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(128, 256, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        # print(feat.shape)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.se=EffectiveSEModule(out_chan)


    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        feat_atten =self.se(feat)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNetV1_Caformer_M36(nn.Module):

    def __init__(self, n_classes, aux_mode='train',use_fp16=False, *args, **kwargs):
        super(BiSeNetV1_Caformer_M36, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(512, 512)
        self.conv_out = BiSeNetOutput(512, 512, n_classes, up_factor=8)
        self.aux_mode = aux_mode
        self.use_fp16=use_fp16
        if self.aux_mode == 'train':
            self.conv_out16 = BiSeNetOutput(256, 128, n_classes, up_factor=8)
            self.conv_out32 = BiSeNetOutput(256, 128, n_classes, up_factor=16)
        # self.init_weight()


    def forward(self, x):
        with autocast(enabled=self.use_fp16,dtype=torch.bfloat16):
            H, W = x.size()[2:]
            feat_cp8, feat_cp16 = self.cp(x)
            feat_sp = self.sp(x)
            feat_fuse = self.ffm(feat_sp, feat_cp8)

            feat_out = self.conv_out(feat_fuse)
            if self.aux_mode == 'train':
                feat_out16 = self.conv_out16(feat_cp8)
                feat_out32 = self.conv_out32(feat_cp16)
                return feat_out, feat_out16, feat_out32
            elif self.aux_mode == 'eval':
                return feat_out,
            elif self.aux_mode == 'pred':
                # feat_out = feat_out.argmax(dim=1)
                feat_out=torch.argmax(feat_out,dim=1)
                feat_out=torch.tensor(feat_out,dtype=torch.float32)
                return feat_out
            else:
                raise NotImplementedError

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


if __name__ == "__main__":
    net = BiSeNetV1_Caformer_M36(19)
    net.eval()
    in_ten = torch.randn(2, 3,224, 224)
    out, out16, out32 = net(in_ten)
    print(out.shape)
    print(out16.shape)
    print(out32.shape)

    net.get_params()
