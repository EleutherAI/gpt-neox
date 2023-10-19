
import torch
from torchvision import transforms, datasets
from mup.shape import set_base_shapes
from torch import nn
from torch.nn import Linear
from mup.layer import MuReadout
from functools import partial
from mup.init import (kaiming_normal_, kaiming_uniform_, normal_,
                         trunc_normal_, uniform_, xavier_normal_,
                         xavier_uniform_)
from torch.nn.modules.conv import _ConvNd

samplers = {
    'default': lambda x: x,
    'const_uniform': partial(uniform_, a=-0.1, b=0.1),
    'const_normal': partial(normal_, std=0.1),
    'const_trunc_normal': partial(trunc_normal_, std=0.1, a=-0.2, b=0.2),
    'xavier_uniform': xavier_uniform_,
    'xavier_normal': xavier_normal_,
    'kaiming_fan_in_uniform': partial(kaiming_uniform_, mode='fan_in'),
    'kaiming_fan_in_normal': partial(kaiming_normal_, mode='fan_in'),
    'kaiming_fan_out_uniform': partial(kaiming_uniform_, mode='fan_out'),
    'kaiming_fan_out_normal': partial(kaiming_normal_, mode='fan_out')
}


def init_model(model, sampler):
    for param in model.parameters():
        if len(param.shape) >= 2:
            sampler(param)
    return model

init_methods = {
    k: partial(init_model, sampler=s) for k, s in samplers.items()
}

def _generate_MLP(width, bias=True, mup=True, batchnorm=False, device='cpu'):
    mods = [Linear(3072, width, bias=bias, device=device),
            nn.ReLU(),
            Linear(width, width, bias=bias, device=device),
            nn.ReLU()
    ]
    if mup:
        mods.append(MuReadout(width, 10, bias=bias, readout_zero_init=False, device=device))
    else:
        mods.append(Linear(width, 10, bias=bias, device=device))
    if batchnorm:
        mods.insert(1, nn.BatchNorm1d(width, device=device))
        mods.insert(4, nn.BatchNorm1d(width, device=device))
    model = nn.Sequential(*mods)
    return model

def generate_MLP(width, bias=True, mup=True, readout_zero_init=True, batchnorm=False, init='default', bias_zero_init=False, base_width=256):
    if not mup:
        model = _generate_MLP(width, bias, mup, batchnorm)
        # set base shapes to model's own shapes, so we get SP
        return set_base_shapes(model, None)
    # it's important we make `model` first, because of random seed
    model = _generate_MLP(width, bias, mup, batchnorm)
    base_model = _generate_MLP(base_width, bias, mup, batchnorm, device='meta')
    set_base_shapes(model, base_model)
    init_methods[init](model)
    if readout_zero_init:
        readout = list(model.modules())[-1]
        readout.weight.data.zero_()
        if readout.bias is not None:
            readout.bias.data.zero_()
    if bias_zero_init:
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    return model


def _generate_CNN(width, bias=True, mup=True, batchnorm=False, device='cpu'):
    mods = [
        nn.Conv2d(3, width, kernel_size=5, bias=bias, device=device),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(width, 2*width, kernel_size=5, bias=bias, device=device),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(2*width*25, width*16, bias=bias, device=device),
        nn.ReLU(inplace=True),
        nn.Linear(width*16, width*10, bias=bias, device=device),
        nn.ReLU(inplace=True),
    ]
    if mup:
        mods.append(MuReadout(width*10, 10, bias=bias, readout_zero_init=False, device=device))
    else:
        mods.append(nn.Linear(width*10, 10, bias=bias, device=device))
    if batchnorm:
        mods.insert(1, nn.BatchNorm2d(width, device=device))
        mods.insert(5, nn.BatchNorm2d(2*width, device=device))
        mods.insert(10, nn.BatchNorm1d(16*width, device=device))
        mods.insert(13, nn.BatchNorm1d(10*width, device=device))
    return nn.Sequential(*mods)

def generate_CNN(width, bias=True, mup=True, readout_zero_init=True, batchnorm=False, init='default', bias_zero_init=False, base_width=8):
    if not mup:
        model = _generate_CNN(width, bias, mup, batchnorm)
        # set base shapes to model's own shapes, so we get SP
        return set_base_shapes(model, None)
    # it's important we make `model` first, because of random seed
    model = _generate_CNN(width, bias, mup, batchnorm)
    base_model = _generate_CNN(base_width, bias, mup, batchnorm, device='meta')
    set_base_shapes(model, base_model)
    init_methods[init](model)
    if readout_zero_init:
        readout = list(model.modules())[-1]
        readout.weight.data.zero_()
        if readout.bias is not None:
            readout.bias.data.zero_()
    if bias_zero_init:
        for module in model.modules():
            if isinstance(module, (nn.Linear, _ConvNd)) and module.bias is not None:
                module.bias.data.zero_()
    return model

def get_lazy_models(arch, widths, mup=True, init='kaiming_fan_in_normal', readout_zero_init=True, batchnorm=True, base_width=None):
    '''if mup is False, then `init`, `readout_zero_init`, `base_width` don't matter.'''
    if arch == 'mlp':
        base_width = base_width or 256
        generate = generate_MLP
    elif arch == 'cnn':
        base_width = base_width or 8
        generate = generate_CNN
    def gen(w):
        def f():
            model = generate(w, mup=mup, init=init, readout_zero_init=readout_zero_init, batchnorm=batchnorm, base_width=base_width)
            return model
        return f
    return {w: gen(w) for w in widths}


def get_train_loader(batch_size, num_workers=0, shuffle=False, train=True, download=False):

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10(root='dataset', train=train,
                                download=download, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers)
