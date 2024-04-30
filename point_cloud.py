from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class EncoderLayerNN(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.transformer1 = nn.Linear(3, 3)
        self.transformer1_2 = nn.Linear(3, 3)
        self.transformer1_3 = nn.Linear(3, 3)

        self.transformer2 = nn.Linear(width, width)
        self.transformer2_drop = nn.Dropout(p=0.2)
        self.transformer2_2 = nn.Linear(width, width - 3)

        self.transformer3 = nn.Linear(width, width)
        self.transformer_drop = nn.Dropout(p=0.2)
        self.transformer3_2 = nn.Linear(width, width)

    def forward(self, tmp, point):
        trans1_res = torch.tanh(self.transformer1(point))
        trans1_res = torch.tanh(self.transformer1_2(trans1_res))
        trans1_res = torch.tanh(self.transformer1_3(trans1_res))

        trans2_res = torch.tanh(self.transformer2(tmp))
        trans2_res = self.transformer2_drop(trans2_res)
        trans2_res = torch.tanh(self.transformer2_2(trans2_res))

        res_trans = self.transformer_drop(torch.tanh(self.transformer3(torch.cat((trans2_res, trans1_res)))))

        return torch.tanh(self.transformer3_2(res_trans))


class EncoderNN(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.layer = EncoderLayerNN(width)
        self.width = width

    def forward(self, pc):
        tmp = Variable(torch.from_numpy(np.array([0.0] * self.width))).float()
        for i in pc[0]:
            tmp = self.layer(tmp, i)

        return tmp


class FinalNN(nn.Module):
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

class PointCloudsNN(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.encoder1 = EncoderNN(width)

        self.encoder2 = EncoderNN(width)

        self.final = FinalNN(width)

    def forward(self, src_pc, dst_pc):
        src_tmp = self.encoder1(src_pc)
        dst_tmp = self.encoder2(dst_pc)

        return self.final(src_tmp, dst_tmp)

