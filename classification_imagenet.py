import torch
import sys
import argparse
from torchvision.models import ResNet34_Weights, ResNet18_Weights, ResNet50_Weights

from dataset import imagenet_dataloader
from utils import *
from ResNet import resnet34, resnet18, resnet50
from torchvision.models import mobilenet_v2
import os

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument('-b','--batch_size', default=100, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed', default=42,type=int,help='seed for initializing training.')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')
parser.add_argument('-arch','--model',default='resnet34',type=str,help='model')
parser.add_argument('-iter', '--iter',default=4000,type=int,metavar='N',help='number of total iterations to run')
parser.add_argument('-t', '--timestep',default=512,type=int,metavar='N',help='snn inference timestep')
parser.add_argument('-dev','--device',default='0',type=str,help='cuda device')
parser.add_argument('-path','--data_path',default="/home/data/public/ImageNet/",type=str,help='your iamgenet data path')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

# load model and evaluate accuracy
if __name__ == "__main__":
    seed_all(args.seed)
    identifier = "imagenet"
    if args.model == "resnet34":
        identifier += "_resnet34"
        model = resnet34()
        weight = ResNet34_Weights.IMAGENET1K_V1
        model.load_state_dict(weight.get_state_dict(progress=True))
    elif args.model == "resnet18":
        identifier += "_resnet18"
        model = resnet18()
        weight = ResNet18_Weights.IMAGENET1K_V1
        model.load_state_dict(weight.get_state_dict(progress=True))
    elif args.model == "resnet50":
        identifier += "_resnet50"
        model = resnet50()
        weight = ResNet50_Weights.IMAGENET1K_V2
        model.load_state_dict(weight.get_state_dict(progress=True))
    elif args.model == "mobilenet":
        model = mobilenet_v2(weights='IMAGENET1K_V1')
        identifier += "_mobilenet"
    else:
        raise AssertionError("No such model!")
    
    identifier += "_iter{}".format(args.iter)
    if args.suffix:
        identifier += "_{}".format(args.suffix)

    train_loader, val_loader = imagenet_dataloader(args.data_path, batchsize=args.batch_size)

    model, cnt = convert_ann_to_snn(model)
    # remove bn
    search_fold_and_remove_bn(model)

    weight_scaling_iter(train_loader, model, "cuda", args.iter)

    acc = snn_inference(val_loader, model, "cuda", args.timestep)
    # print(acc)
    np.save('./%s_t%d.npy'%(identifier, args.timestep), acc)