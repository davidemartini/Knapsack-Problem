
import random 
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd 


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
        self.L1 = nn.Linear(self.input_size, 50)
        self.L2 = nn.Linear(100, 32)
        self.Lhid = nn.Linear(32, 50)
        self.Lout = nn.Linear(50, self.output_size)

    def concat(self, input, hidden):
        if self.gpu:
            output = torch.rand(1, 2*50).cuda()
        else:
            output = torch.rand(1, 2*50)
        for i in range(input.shape[1]):
            output[0, i] = input[0, i]
            output[0, i+50] = hidden[0, i]
        return output

    # For function that will activate neurons and perform forward propagation
    def forward(self, state, hidden):
        if self.gpu:
            level = self.L1(state).cuda()
            combined = self.concat(level, hidden).cuda()
            level = self.L2(combined).cuda()
            hidden = self.Lhid(level).cuda()
            output = self.Lout(hidden).cuda()
            output = torch.sigmoid(output).cuda()
        else:
            level = self.L1(state)
            combined = self.concat(level, hidden)
            level = self.L2(combined)
            hidden = self.Lhid(level)
            output = self.Lout(hidden)
            output = torch.sigmoid(output)


        return output, hidden

    def initHidden(self):
        hid = []
        if self.gpu:
            hid.append(torch.rand(self.input_size, 50).cuda())
            hid.append(torch.rand(100, 32).cuda())
            hid.append(torch.rand(32, 50).cuda())
            hid.append(torch.rand(50, self.output_size).cuda())
        else:
            hid.append(torch.rand(self.input_size, 50))
            hid.append(torch.rand(50, 32))
            hid.append(torch.rand(32, 50))
            hid.append(torch.rand(50, self.output_size))
        return hid
