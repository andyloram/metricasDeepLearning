import torch
import torch.nn as nn
import nnParts as nnp
import torch.nn.functional as F


class Dasnet(nn.Module):

    def __init__(self):
        super(Dasnet, self).__init__()

        self.first_age = nnp.DoubleConv(1, 64, 3, max_pool=2)
        self.first_sex = nnp.DoubleConv(1, 64, 3, max_pool=2)

        self.sec_age = nnp.DoubleConv(128, 64, 3, max_pool=2)
        self.sec_sex = nnp.DoubleConv(64, 64, 3, max_pool=2)

        self.third_age = nnp.DoubleConv(128, 128, 3, max_pool=2)
        self.third_sex = nnp.DoubleConv(64, 128, 3, max_pool=2)

        self.fourth_age = nnp.DoubleConv(256, 256, 3, max_pool=2)
        self.fourth_sex = nnp.DoubleConv(128, 256, 3, max_pool=2)

        self.fifth_age = nnp.SingleConv(512, 512, 3)
        self.fifth_sex = nnp.SingleConv(256, 512, 3)

        self.sixth_age = nnp.SingleConv(512, 512, 2)
        self.sixth_sex = nnp.SingleConv(512, 512, 2)

        self.seventh_age = nn.Linear(1024, 1)
        self.seventh_sex = nn.Linear(512, 1)


    def forward(self, x):
        age = self.first_age(x)
        sex = self.first_sex(x).float()

        age = torch.cat((age,sex), 1)
        age = self.sec_age(age)
        sex = self.sec_sex(sex)

        age = torch.cat((age, sex), 1)
        age = self.third_age(age)
        sex = self.third_sex(sex)

        age = torch.cat((age, sex), 1)
        age = self.fourth_age(age)
        sex = self.fourth_sex(sex)

        age = torch.cat((age, sex), 1)
        age = self.fifth_age(age)
        age = self.sixth_age(age)

        sex = self.fifth_sex(sex)
        sex = self.sixth_sex(sex)

        age = F.avg_pool2d(age, kernel_size=(1, 9))
        sex = F.avg_pool2d(sex, kernel_size=(1, 9))

        age = torch.cat((age, sex), 1)
        age = age.view(age.size(0), -1)
        age = self.seventh_age(age)

        sex = sex.view(sex.size(0), -1)
        sex = self.seventh_sex(sex)
        sex = torch.sigmoid(sex)

        return [age, sex]

def lt(tensor):
    return tensor.type(torch.LongTensor)