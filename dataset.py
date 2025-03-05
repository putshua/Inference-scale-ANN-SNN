from textwrap import fill
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os


def imagenet_dataloader(dataroot, batchsize):
    trans_t = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    
    trans = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

    train_data = datasets.ImageFolder(root=os.path.join(dataroot, 'train'), transform=trans_t)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader =DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True)

    test_data = datasets.ImageFolder(root=os.path.join(dataroot, 'val'), transform=trans)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=4) 
    return train_dataloader, test_dataloader