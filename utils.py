import torch
from tqdm import tqdm
from layers import *
import torch.nn as nn
import numpy as np
import random
import os

def snn_inference(dataloader, model, device, timesteps, st=16):
    for module in model.modules():
        if isinstance(module, (SpikingNeuron, SpikingNeuron2d, SpikingNeuron4d)):
            module.mode = "snn"
    tot = np.zeros(timesteps)
    sops = np.zeros(timesteps)
    model.eval()
    model.to(device)
    length = 0
    length2 = 0
    with torch.no_grad():
        for img, label in tqdm(dataloader):
            img = img.to(device)
            label = label.to(device)
            outs = []
            for t in range(timesteps):
                outs.append(model(img))

            out_spike = torch.stack(outs, 0)
            out_spike[:st] = torch.cumsum(out_spike[:st], dim=0)
            out_spike[st:] = torch.cumsum(out_spike[st:], dim=0)
            reset_model(model)
            length += len(label)
            for i in range(timesteps):
                tot[i] += (label==out_spike[i].max(1)[1]).sum().data
    return tot/length

def evaluate(test_dataloader, model, device):
    tot = 0.
    model.eval()
    model.to(device)
    length = 0
    with torch.no_grad():
        for img, label in tqdm(test_dataloader):
            img = img.to(device)
            label = label.to(device)
            out = model(img).logits
            # loss = loss_fn(out, label)
            length += len(label)    
            tot += (label==out.max(1)[1]).sum().data
    return tot/length

def convert_ann_to_snn(model, cnt=0):
    prev_module = None
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name], cnt = convert_ann_to_snn(module, cnt)
        if isinstance(module, (nn.ReLU, nn.ReLU6)):
            if isinstance(model._modules[prev_module], nn.BatchNorm2d):
                print("Convert Layer {}, ReLU to SpkingNeuron4d".format(cnt))
                model._modules[name] = SpikingNeuron4d(num_features=model._modules[prev_module].num_features)
            elif isinstance(model._modules[prev_module], nn.Linear):
                print("Convert Layer {}, ReLU to SpkingNeuron2d".format(cnt))
                model._modules[name] = SpikingNeuron2d(num_features=model._modules[prev_module].out_features)
            else:
                raise AssertionError(prev_module)
            cnt += 1
        if isinstance(module, nn.MaxPool2d) and isinstance(model._modules[prev_module], SpikingNeuron4d):
            print("Using prev spike maxpooling")
            prev_feature = model._modules[prev_module].num_features
            model._modules[prev_module] = module
            model._modules[name] = SpikingNeuron4d(num_features=prev_feature)
        elif isinstance(module, nn.MaxPool2d) and isinstance(model._modules[prev_module], SpikingNeuron2d):
            print("Using prev spike maxpooling")
            prev_feature = model._modules[prev_module].num_features
            model._modules[prev_module] = module
            model._modules[name] = SpikingNeuron2d(num_features=prev_feature)
        
        prev_module = name
    return model, cnt

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def reset_model(model):
    for module in model.modules():
        if isinstance(module, (SpikingNeuron, SpikingNeuron2d, SpikingNeuron4d)):
            module.reset()
    return

def weight_scaling_iter(dataloader, model, device, iter):
    tot = 0.
    length = 0.
    maxlen = 10
    buffer = torch.zeros(maxlen)
    losses = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for i, (img, label) in (enumerate(dataloader)):
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            ss = 0.
            layers = 0
            for module in model.modules():
                if isinstance(module, (SpikingNeuron, SpikingNeuron2d, SpikingNeuron4d)):
                    ss += module.delta
                    layers += 1
            losses.append(ss/layers)
            length += len(label)    
            tot = (label==out.max(1)[1]).sum().data
            buffer[i%maxlen] = tot
            if i<maxlen:
                acc = buffer.sum().item()/((i+1)*len(label))
                print("iter: {}, Acc: {}, Delta: {}".format(i, acc, ss))
            else:
                acc = buffer.sum().item()/(maxlen*len(label))
                print("iter: {}, Acc: {}, Delta: {}".format(i, acc, ss))
            if i > iter:
                break
    return losses


def _fold_bn(conv_module, bn_module, avg=False):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module, avg=False):
    w, b = _fold_bn(conv_module, bn_module, avg)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2


def is_absorbing(m):
    return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)


def search_fold_and_remove_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if isinstance(m, nn.BatchNorm2d) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
            setattr(model, n, Dummy())
        elif is_absorbing(m):
            prev = m
        else:
            prev = search_fold_and_remove_bn(m)
    return prev