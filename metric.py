import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class CosFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        return output * self.s


class ArcFace(nn.Module):
    def __init__(self, embedding_size, class_num, s=30.0, m=0.50,):
        super().__init__()
        self.embedding_size = embedding_size
        self.out_features = class_num
        self.s = s
        self.weights = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.xavier_uniform_(self.weights)


        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)# ???
        self.th = math.cos(math.pi - m) # make sure margin should be (0 - pi)
        self.mm = math.sin(math.pi - m) * m # ???


    def forward(self, x):
        cosine = F.linear(F.normalize(x), F.normalize(self.weights))
        sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()

        # justify angle m is 
        phi = cosine * self.cos_m - sine * self.sin_m



