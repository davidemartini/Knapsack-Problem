import torch
import torch.nn as nn


# Creating the architecture of the Neural Network
class Network(nn.Module):

    def __init__(self, input_size, output_size, gpu):
        super(Network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gpu = gpu
        self.lhid1 = nn.Linear(self.input_size, 1024)
        self.lhid2 = nn.Linear(1024, 950)
        self.lhid3 = nn.Linear(950, 900)
        self.lhid4 = nn.Linear(900, 850)
        self.lhid5 = nn.Linear(850, 800)
        self.lhid6 = nn.Linear(800, 750)
        self.lhid7 = nn.Linear(750, 700)
        self.lhid8 = nn.Linear(700, 650)
        self.lhid9 = nn.Linear(650, 600)
        self.lhid10 = nn.Linear(600, 550)
        self.drop = nn.Dropout(p=0.6)
        self.lhid11 = nn.Linear(550, 500)
        self.lhid12 = nn.Linear(500, 450)
        self.lhid13 = nn.Linear(450, 400)
        self.lhid14 = nn.Linear(400, 375)
        self.lhid15 = nn.Linear(375, 350)
        self.lhid16 = nn.Linear(350, 325)
        self.lhid17 = nn.Linear(325, 300)
        self.lhid18 = nn.Linear(300, 275)
        self.lhid19 = nn.Linear(275, 250)
        self.lhid20 = nn.Linear(250, 225)
        self.lhid21 = nn.Linear(225, 200)
        self.lout = nn.Linear(200, self.output_size)

    # For function that will activate neurons and perform forward propagation
    def forward(self, state):
        if self.gpu:
            level = self.lhid1(state).cuda()
            level = self.lhid2(level).cuda()
            level = self.lhid3(level).cuda()
            level = self.lhid4(level).cuda()
            level = self.lhid5(level).cuda()
            level = self.lhid6(level).cuda()
            level = self.lhid7(level).cuda()
            level = self.lhid8(level).cuda()
            level = self.lhid9(level).cuda()
            level = self.lhid10(level).cuda()
            # level = self.drop(level).cuda()
            level = self.lhid11(level).cuda()
            level = self.lhid12(level).cuda()
            level = self.lhid13(level).cuda()
            level = self.lhid14(level).cuda()
            level = self.lhid15(level).cuda()
            level = self.lhid16(level).cuda()
            level = self.lhid17(level).cuda()
            level = self.lhid18(level).cuda()
            level = self.lhid19(level).cuda()
            level = self.lhid20(level).cuda()
            level = self.lhid21(level).cuda()
            output = self.lout(level).cuda()
            output = torch.sigmoid(output).cuda()
        else:
            level = self.lhid1(state)
            level = self.lhid2(level)
            level = self.lhid3(level)
            level = self.lhid4(level)
            level = self.lhid5(level)
            level = self.lhid6(level)
            level = self.lhid7(level)
            level = self.lhid8(level)
            level = self.lhid9(level)
            level = self.lhid10(level)
            # level = self.drop(level)
            level = self.lhid11(level)
            level = self.lhid12(level)
            level = self.lhid13(level)
            level = self.lhid14(level)
            level = self.lhid15(level)
            level = self.lhid16(level)
            level = self.lhid17(level)
            level = self.lhid18(level)
            level = self.lhid19(level)
            level = self.lhid20(level)
            level = self.lhid21(level)
            output = self.lout(level)
            output = torch.sigmoid(output)
        return output

    def initHidden(self):
        hid = []
        if self.gpu:
            hid.append(torch.rand(self.input_size, 1024).cuda())
            hid.append(torch.rand(1024, 1000).cuda())
            hid.append(torch.rand(1000, 800).cuda())
            hid.append(torch.rand(800, 600).cuda())
            hid.append(torch.rand(600, 400).cuda())
            hid.append(torch.rand(400, 350).cuda())
            hid.append(torch.rand(350, 325).cuda())
            hid.append(torch.rand(300, 275).cuda())
            hid.append(torch.rand(275, 250).cuda())
            hid.append(torch.rand(250, self.output_size).cuda())
        else:
            hid.append(torch.rand(self.input_size, 1024))
            hid.append(torch.rand(1024, 1000))
            hid.append(torch.rand(1000, 800))
            hid.append(torch.rand(800, 600))
            hid.append(torch.rand(600, 400))
            hid.append(torch.rand(400, 350))
            hid.append(torch.rand(350, 325))
            hid.append(torch.rand(300, 275))
            hid.append(torch.rand(275, 250))
            hid.append(torch.rand(250, self.output_size))
        return hid
