
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
    def __init__(self, input_size, batch_size, output_size, gpu, lr, h1, h2, h3, h4, h5): #[self,input neuroner, output neuroner]
        super(Network, self).__init__() #inorder to use modules in torch.nn
        # Input and output neurons
        self.input_size = input_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.h4 = h4
        self.h5 = h5
        self.lhid1 = nn.Linear(self.batch_size * self.input_size, self.h1)
        self.lhid2 = nn.Linear(self.h1, self.h2)
        self.lhid3 = nn.Linear(self.h2, self.h3)
        self.lhid4 = nn.Linear(self.h3, self.h4)
        self.lhid5 = nn.Linear(self.h4, self.h5)
        self.lout = nn.Linear(self.h5, self.output_size)

    # For function that will activate neurons and perform forward propagation
    def forward(self, state):
        # rectifier function
        x = self.lhid1(state)
        #x = F.relu(x)
        x = self.lhid2(x)
        x = self.lhid3(x)
        x = self.lhid4(x)
        x = self.lhid5(x)
        x = self.lout(x)
        x = torch.sigmoid(x)
        return x

    def initHidden(self):
        return torch.random(self.batch_size * self.input_size, self.h1), torch.random(self.h1, self.h2), torch.random(self.h2, self.h3), torch.random(self.h3, self.h4), torch.random(self.h4, self.h5)