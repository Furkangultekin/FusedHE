import torch.nn as nn
import torchvision.models as models


from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmcv.runner import load_checkpoint


class Resnet(nn.Module):
    def __init__(self, pre_trained=False) -> None:
        super(Resnet, self).__init__()

        self.resnet = models.resnet101(pretrained=pre_trained)
    
    def forward(self, x):
        layer_outs = []
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        layer_outs.append(x)
        x = self.resnet.layer2(x)
        layer_outs.append(x)
        x = self.resnet.layer3(x)
        layer_outs.append(x)
        x = self.resnet.layer4(x)
        layer_outs.append(x)

        return layer_outs