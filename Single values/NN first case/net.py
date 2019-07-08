
import random # random samples from different batches (experience replay)
import os # For loading and saving brain
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd # Conversion from tensor (advanced arrays) to avoid all that contains a gradient


# Creating the architecture of the Neural Network
class Network(nn.Module): #inherinting from nn.Module

    #Self - refers to the object that will be created from this class
    #     - self here to specify that we're referring to the object
    def __init__(self, input_size, output_size, gpu): #[self,input neuroner, output neuroner]
        super(Network, self).__init__() #inorder to use modules in torch.nn
        # Input and output neurons
        self.input_size = input_size
        self.output_size = output_size
        self.gpu = gpu
        self.lhid1 = nn.Linear(self.input_size, 512)
        self.lhid2 = nn.Linear(512, 256)
        self.lhid3 = nn.Linear(256, 128)
        self.lhid4 = nn.Linear(128, 64)
        self.lhid5 = nn.Linear(64, 32)
        self.lhid6 = nn.Linear(32, 16)
        self.lhid7 = nn.Linear(16, 8)
        self.lhid8 = nn.Linear(8, 4)
        self.lhid9 = nn.Linear(4, 2)
        self.lout = nn.Linear(2, self.output_size)

    # For function that will activate neurons and perform forward propagation
    def forward(self, state):
        # rectifier function
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
            output = self.lout(level)
            output = torch.sigmoid(output)
        return output

    def initHidden(self):
        hid = []
        if self.gpu:
            hid.append(torch.rand(self.input_size, 512).cuda())
            hid.append(torch.rand(512, 256).cuda())
            hid.append(torch.rand(256, 128).cuda())
            hid.append(torch.rand(128, 64).cuda())
            hid.append(torch.rand(64, 32).cuda())
            hid.append(torch.rand(32, 16).cuda())
            hid.append(torch.rand(16, 8).cuda())
            hid.append(torch.rand(8, 4).cuda())
            hid.append(torch.rand(4, 2).cuda())
            hid.append(torch.rand(2, self.output_size).cuda())
        else:
            hid.append(torch.rand(self.input_size, 512))
            hid.append(torch.rand(512, 256))
            hid.append(torch.rand(256, 128))
            hid.append(torch.rand(128, 64))
            hid.append(torch.rand(64, 32))
            hid.append(torch.rand(32, 16))
            hid.append(torch.rand(16, 8))
            hid.append(torch.rand(8, 4))
            hid.append(torch.rand(4, 2))
            hid.append(torch.rand(2, self.output_size))
        return hid
