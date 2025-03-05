import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class Dummy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class SpikingNeuron(nn.Module):
    def __init__(self, c, mode="ann"):
        super(SpikingNeuron, self).__init__()
        self.register_buffer("thre", torch.tensor(0.))
        self.mode = mode
        self.T = 0
        self.mem = 0.
        self.delta = 0.
        self.buffer = None
        self.total_count = 0
        self.spike_count = 0
        self.op = 0.
        self.register_buffer("c", c)
        self.register_buffer("macs", torch.tensor(0.))

    def optimize(self, x):
        ub = self.thre
        self.thre += self.c * 2 * ((x - ub) * (x > ub).float()).mean()
        y = torch.clamp(x, 0, self.thre.item())
        return y
    
    def robust_norm(self, x):
        x = F.relu(x)
        q = x.reshape(-1).clone().detach().quantile(0.99, interpolation='nearest')
        self.thre = max(q, self.thre)
        return x

    def forward(self, x):
        if self.mode == "snn":
            if self.T == 0:
                self.mem = 0.5 * self.thre
            self.mem = self.mem + x
            spike_count = ((self.mem -self.thre) > 0).sum().item()
            total_count = torch.ones_like(x).sum().item()
            self.op += spike_count/total_count * self.macs
            x = ((self.mem -self.thre) > 0) * self.thre
            self.mem = self.mem - x
            self.T += 1
        elif self.mode == "clip":
            x = torch.clamp(x, 0, self.thre)
        elif self.mode == "robust_norm":
            x = self.robust_norm(x)
        else:
            y = self.optimize(x)
            self.delta = torch.norm(y-x,p="fro").item()
            x = y
        return x

    def reset(self):
        self.mem = 0.
        self.T = 0
        self.op = 0.
        return


class SpikingNeuron2d(SpikingNeuron):
    def __init__(self, num_features, c=torch.tensor(1.), mode="ann"):
        super(SpikingNeuron2d, self).__init__(c, mode)
        self.register_buffer("thre", torch.zeros((1, num_features)))
        self.num_features = num_features

    def optimize(self, x):
        ub = self.thre
        assert len(x.shape) == 2
        self.thre += self.c * (2*(x - ub) * (x > ub)).mean(0, keepdim=True)
        self.delta = ((x - ub) * (x > ub).float()).mean().item()

        x = F.relu(x, inplace='True')
        x = self.thre - x
        x = F.relu(x, inplace='True')
        x = self.thre - x
        return x
    
class SpikingNeuron4d(SpikingNeuron):
    def __init__(self, num_features, c=torch.tensor([1.]), mode="ann"):
        super(SpikingNeuron4d, self).__init__(c, mode)  
        self.register_buffer("thre", torch.zeros((1, num_features, 1, 1)))
        self.num_features = num_features

    def optimize(self, x):
        ub = self.thre
        assert len(x.shape) == 4
        self.thre += self.c * (2*(x - ub) * (x > ub)).mean((0,2,3), keepdim=True)
        self.delta = ((x - ub) * (x > ub).float()).mean().item()

        x = F.relu(x, inplace='True')
        x = self.thre - x
        x = F.relu(x, inplace='True')
        x = self.thre - x
        return x