import os, sys, pickle, math
from batchup.datasets import dataset
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models, torchvision.models.resnet as resnet, torchvision.transforms as transforms

_ARCH_REGISTRY = {}


def architecture(name):
    """
    Decorator to register an architecture;

    Use like so:

    >>> @architecture('my_architecture')
    ... def build_network(n_classes, gaussian_noise_std):
    ...     # Build network
    ...     return dict(net=last_layer)
    """
    def decorate(fn):
        _ARCH_REGISTRY[name] = fn
        return fn
    return decorate


def get_build_fn_for_architecture(arch_name):
    """
    Get network building function and expected sample shape:

    For example:
    >>> fn = get_build_fn_for_architecture('my_architecture')
    """
    return _ARCH_REGISTRY[arch_name]



def _unpickle_from_path(path):
    # Oh... the joys of Py2 vs Py3
    with open(path, 'rb') as f:
        if sys.version_info[0] == 2:
            return pickle.load(f)
        else:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            return u.load()




#
#
# CUSTOM RESNET CLASS
#
#



class DomainAdaptModule (nn.Module):
    def _init_bn_layers(self, layers):
        self._bn_layers = layers
        self._bn_src_values = []
        self._bn_tgt_values = []

    def bn_save_source(self):
        self._bn_src_values = []
        for layer in self._bn_layers:
            self._bn_src_values.append(layer.running_mean.clone())
            self._bn_src_values.append(layer.running_var.clone())

    def bn_restore_source(self):
        for i, layer in enumerate(self._bn_layers):
            layer.running_mean.copy_(self._bn_src_values[i*2 + 0])
            layer.running_var.copy_(self._bn_src_values[i*2 + 1])

    def bn_save_target(self):
        self._bn_tgt_values = []
        for layer in self._bn_layers:
            self._bn_tgt_values.append(layer.running_mean.clone())
            self._bn_tgt_values.append(layer.running_var.clone())

    def bn_restore_target(self):
        for i, layer in enumerate(self._bn_layers):
            layer.running_mean.copy_(self._bn_tgt_values[i*2 + 0])
            layer.running_var.copy_(self._bn_tgt_values[i*2 + 1])



class ResNet(DomainAdaptModule):
    def __init__(self, block, layers, num_classes=1000, avgpool_size=7, use_dropout=False):
        self.inplanes = 64
        self.use_dropout = use_dropout
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(avgpool_size)
        self.new_fc5 = nn.Linear(512 * block.expansion, 512)
        if use_dropout:
            self.new_drop_fc5 = nn.Dropout2d(0.5)
        else:
            self.new_drop_fc5 = None
        self.new_fc6 = nn.Linear(512, num_classes)

        bn_layers = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                bn_layers.append(m)

        self._init_bn_layers(bn_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.new_fc5(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.new_drop_fc5(x)
        x = self.new_fc6(x)
        # x = F.softmax(x)

        return x

def resnet50(num_classes=1000, avgpool_size=7, use_dropout=False, pretrained=True):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes, avgpool_size=avgpool_size,
                   use_dropout=use_dropout)

    if pretrained:
        state_dict = resnet.model_zoo.load_url(resnet.model_urls['resnet50'])

        current_state = model.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if not key.startswith('fc.'):
                current_state[key] = state_dict[key]

        model.load_state_dict(current_state)
    return model

def resnet101(num_classes=1000, avgpool_size=7, use_dropout=False, pretrained=True):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(resnet.Bottleneck, [3, 4, 23, 3], num_classes=num_classes, avgpool_size=avgpool_size,
                   use_dropout=use_dropout)

    if pretrained:
        state_dict = resnet.model_zoo.load_url(resnet.model_urls['resnet101'])

        current_state = model.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if not key.startswith('fc.'):
                current_state[key] = state_dict[key]

        model.load_state_dict(current_state)
    return model


def resnet152(num_classes=1000, avgpool_size=7, use_dropout=False, pretrained=True):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(resnet.Bottleneck, [3, 8, 36, 3], num_classes=num_classes, avgpool_size=avgpool_size,
                   use_dropout=use_dropout)

    if pretrained:
        state_dict = resnet.model_zoo.load_url(resnet.model_urls['resnet152'])

        current_state = model.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if not key.startswith('fc.'):
                current_state[key] = state_dict[key]

        model.load_state_dict(current_state)
    return model


@architecture('resnet50')
def build_resnet50(n_classes, img_size, use_dropout, pretrained):
    return resnet50(num_classes=n_classes, avgpool_size=img_size//32, use_dropout=use_dropout, pretrained=pretrained)


@architecture('resnet101')
def build_resnet101(n_classes, img_size, use_dropout, pretrained):
    return resnet101(num_classes=n_classes, avgpool_size=img_size//32, use_dropout=use_dropout, pretrained=pretrained)


@architecture('resnet152')
def build_resnet152(n_classes, img_size, use_dropout, pretrained):
    return resnet152(num_classes=n_classes, avgpool_size=img_size//32, use_dropout=use_dropout, pretrained=pretrained)



def robust_binary_crossentropy(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = -pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))

def bugged_cls_bal_bce(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))

def log_cls_bal(pred, tgt):
    return -torch.log(pred + 1.0e-6)

def get_cls_bal_function(name):
    if name == 'bce':
        return robust_binary_crossentropy
    elif name == 'log':
        return log_cls_bal
    elif name == 'bug':
        return bugged_cls_bal_bce