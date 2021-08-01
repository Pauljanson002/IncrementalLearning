from .ConvNet import convnet4
from .ViTLite import ViTLite, vit_lite_100_feature_extractor
from .cct import cct7_h
from .ResNet import resnet18_cbam

model_pool = [
    'convnet4',
]
model_dict = {
    'resnet18_3': '',
    'convnet4': convnet4,
}


def get_feature_extractor(name):
    if name == 'resnet':
        return resnet18_cbam()
    elif name == 'vit_lite':
        return vit_lite_100_feature_extractor()
    elif name == 'cct7_h':
        return cct7_h()
    else:
        raise NotImplementedError("The required model is not implemented")
