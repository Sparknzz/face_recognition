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


class ArcFace1(nn.Module):
    def __init__(self, embedding_size, class_num, s=30.0, m=0.50,):
        super().__init__()
        self.embedding_size = embedding_size
        self.out_features = class_num
        self.s = s
        self.weights = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.xavier_uniform_(self.weights)

        self.m = m

        # self.cos_m = math.cos(m)
        # self.sin_m = math.sin(m)
        # self.th = math.cos(math.pi - m) # make sure margin should be (0 - pi)
        # self.mm = math.sin(math.pi - m) * m


    def forward(self, x, label):
        # print(x.shape) # 64, 512
        # print(label.shape) # 64
        cosine = F.linear(F.normalize(x), F.normalize(self.weights))
        cosine_ij = torch.gather(cosine, dim=-1, index=label.view(-1, 1)) # 64, 1
        
        theta_ij = torch.acos(cosine_ij)
        new_theta_ij = theta_ij + self.m

        new_cosine_ij = torch.cos(new_theta_ij)

        diff_ij = new_cosine_ij - cosine_ij

        gt_one_hot = torch.zeros(cosine.size(), device='cuda')
        gt_one_hot = gt_one_hot.scatter_(1, label.view(-1,1).long(), 1)

        # new_cosine_ij =  cosine(theta_ij+self.m) = diff + cosine_ij
        # diff = new_cosine_ij - cosine_ij


        out = gt_one_hot * diff_ij + cosine

        return out
    # below is insight face mxnet implementation of arcface
    #     zy = mx.sym.pick(fc7, gt_label, axis=1)
    #     cos_t = zy/s
    #     t = mx.sym.arccos(cos_t)
    #     if config.loss_m1!=1.0:
    #       t = t*config.loss_m1
    #     if config.loss_m2>0.0:
    #       t = t+config.loss_m2
    #     body = mx.sym.cos(t)
    #     if config.loss_m3>0.0:
    #       body = body - config.loss_m3
    #     new_zy = body*s
    #     diff = new_zy - zy
    #     diff = mx.sym.expand_dims(diff, 1)
    #     gt_one_hot = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = 1.0, off_value = 0.0)
    #     body = mx.sym.broadcast_mul(gt_one_hot, diff)
    #     fc7 = fc7+body

class ArcFace(nn.Module):
    def __init__(self, embedding_size, class_num, s=30.0, m=0.50, easy_margin=False):
        super(ArcFace, self).__init__()
        self.in_features = embedding_size
        self.out_features = class_num
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output
