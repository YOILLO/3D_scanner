from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class EncoderLayerСNN(nn.Module):
    def __init__(self, width, length):
        super().__init__()

        self.transformer_conv1 = nn.Conv1d(length, length, 3, stride=2, dilation=1, bias=False)

        self.enc1 = nn.Linear(length, length // 2)
        self.enc1_drop = nn.Dropout(p=0.2)
        self.enc2 = nn.Linear(length // 2, length // 3)
        self.enc2_drop = nn.Dropout(p=0.2)
        self.enc3 = nn.Linear(length // 3, length // 3)
        self.enc3_drop = nn.Dropout(p=0.2)

        self.dec1 = nn.Linear(length // 3, length // 2)
        self.dec1_drop = nn.Dropout(p=0.2)
        self.dec2 = nn.Linear(length // 2, length)
        self.dec2_drop = nn.Dropout(p=0.2)

        self.transformer1 = nn.Linear(length, width * 2)
        self.transformer2 = nn.Linear(width * 2,  width)
        self.transformer3 = nn.Linear(width, width//2)
        self.transformer_drop1 = nn.Dropout(p=0.2)
        self.transformer4 = nn.Linear(width, width//2)
        self.transformer_drop2 = nn.Dropout(p=0.2)
        self.transformer5 = nn.Linear(width, width)
        self.transformer_drop3 = nn.Dropout(p=0.2)

    def forward(self, tmp, points):
        tmp1 = self.transformer_drop2(self.transformer4(tmp))

        point_cnd = self.transformer_conv1(points)
        point_cnd = torch.tanh(point_cnd)

        point_cnd = torch.flatten(point_cnd)

        point_cnd = torch.tanh(self.enc1_drop(self.enc1(point_cnd)))
        point_cnd = torch.tanh(self.enc2_drop(self.enc2(point_cnd)))
        point_cnd = torch.tanh(self.enc3_drop(self.enc3(point_cnd)))

        point_cnd = torch.tanh(self.dec1_drop(self.dec1(point_cnd)))
        point_cnd = torch.tanh(self.dec2_drop(self.dec2(point_cnd)))

        point_cnd2 = torch.tanh(self.transformer1(point_cnd))
        point_cnd2 = torch.tanh(self.transformer2(point_cnd2))
        point_cnd2 = torch.tanh(self.transformer_drop2(self.transformer3(point_cnd2)))
        return torch.tanh(self.transformer_drop3(self.transformer5(torch.cat((point_cnd2, tmp1)))))

class EncoderСNN(nn.Module):
    def __init__(self, width, length):
        super().__init__()

        self.layer = EncoderLayerСNN(width, length)
        self.width = width
        self.length = length

    def forward(self, pc):
        tmp = Variable(torch.from_numpy(np.array([0.0] * self.width))).float()
        splited_pc = torch.split(pc, self.length, 1)
        for i in splited_pc:
            tmp = self.layer(tmp, i)

        return tmp

class FinalСNN(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.final1 = nn.Linear(width * 2, width * 2)
        self.final_drop1 = nn.Dropout(p=0.3)
        self.final2 = nn.Linear(width * 2, width * 2)
        self.final_drop2 = nn.Dropout(p=0.3)
        self.final3 = nn.Linear(width * 2, width * 2)
        self.final_drop3 = nn.Dropout(p=0.3)
        self.final4 = nn.Linear(width * 2, width * 2)
        self.final_drop4 = nn.Dropout(p=0.3)

        self.rot1 = nn.Linear(width, width)
        self.rot2 = nn.Linear(width, 5)
        self.rot3 = nn.Linear(5, 3)

        self.trans1 = nn.Linear(width, width)
        self.trans2 = nn.Linear(width, 5)
        self.trans3 = nn.Linear(5, 3)
        self.width = width

    def forward(self, src_tmp, dst_tmp):
        tmp = self.final_drop1(torch.tanh(self.final1(torch.cat((src_tmp, dst_tmp)))))
        tmp = self.final_drop2(torch.tanh(self.final2(tmp)))
        tmp = self.final_drop3(torch.tanh(self.final3(tmp)))
        tmp = self.final_drop4(torch.tanh(self.final4(tmp)))

        rot, trans = torch.split(tmp, self.width)

        rot = torch.tanh(self.rot1(rot))
        rot = torch.tanh(self.rot2(rot))
        rot = self.rot3(rot)
        trans = torch.tanh(self.trans1(trans))
        trans = torch.tanh(self.trans2(trans))
        trans = self.trans3(trans)

        return torch.cat((rot, trans))

class PointCloudsСNN(nn.Module):
    def __init__(self, width, length):
        super().__init__()

        self.encoder1 = EncoderСNN(width, length)

        self.encoder2 = EncoderСNN(width, length)

        self.final = FinalСNN(width)

    def forward(self, src_pc, dst_pc):
        src_tmp = self.encoder1(src_pc)
        dst_tmp = self.encoder2(dst_pc)

        return self.final(src_tmp, dst_tmp)

