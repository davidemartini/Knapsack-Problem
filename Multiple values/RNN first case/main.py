import sys
import os
import data as dt
from timeit import default_timer as timer
import net as net
import numpy as np
import torch.nn as nn
import torch.optim as optim # for using stochastic gradient descent
from torch.autograd import Variable
import torch
#import matplotlib as mpl
#mpl.use('tkagg')
#import matplotlib.pyplot as plt
import smtplib


def write_all(data, path, file, measure, type):
    if path != '':
        element = os.listdir(path)
    else:
        element = ['plot']
    if 'plot' not in element:
        os.system('mkdir plot')
    if measure == 'loss':
        fout = open('plot/'+file, 'w')
        fout.write('LOSS FUNCTION\n')
        for i in range(len(data)):
            fout.write(str(data[i])+" ")
        fout.write('\n')
        fout.close()
    elif measure == 'accuracy':
        fout = open('plot/'+file, 'a')
        if type == 'val':
            fout.write('ACCURACY VALIDATION\n')
        elif type == 'test':
            fout.write('ACCURACY TEST\n')
        for i in range(len(data)):
            fout.write(str(data[i])+" ")
        fout.write('\n')
        fout.close()
    elif measure == 'error':
        fout = open('plot/'+file, 'a')
        if type == 'val':
            fout.write('ERROR VALIDATION\n')
        elif type == 'test':
            fout.write('ERROR TEST\n')
        for i in range(len(data)):
            for j in range(len(data[i])):
                fout.write(str(data[i][j])+" ")
            fout.write('\n')
        fout.write('\n')
        fout.close()


def save_model(model, path):
    torch.save(model.state_dict(), path)


def read_params(path):
    verbose = True
    gpu = False
    path = './sorted/'
    f = 'rnnparam.txt'
    m = 'rnnparam.pt'
    params = sys.argv
    if len(params) > 1:
        for i in range(len(params)):
            if params[i] == "--path":
                if len(params) > i+1:
                    path = params[i+1]
                else:
                    usage(True)
                    sys.exit()
            elif params[i] == "--v":
                verbose = True
            elif params[i] == "--help" or params[i] == "-h":
                usage(False)
                sys.exit()
            elif params[i] == "--gpu":
                gpu = True
    file = params[0].split('.')
    if (file[1] == 'py'):
        name = file[0].split('/')
        f = name[-1]+".txt"
        m = name[-1]+".pt"
    return path, verbose, f, m, gpu


def main(sourcep):
    # read all arguments
    path, verbose, file, modelf, gpu = read_params(sourcep)
    if verbose:
        print('###\tStart reading data')
    start = timer()
    train, validation, test = dt.read_data(path)
    end = timer()
    if verbose:
        print('###\tTime for reading data: %.2f sec' % (end - start))
    return train, validation, test, verbose, path, file, modelf, gpu


# order to execute
torch.cuda.init()
path = './sorted/'
lossfile = 'losses.txt'
train, validation, test, verbose, path, file, modelf, gpu = main(path)
n_epochs = int(len(train))
n_iters = train[0].n_item
input_size = 400
output_size = 3 # fixed
lr = 0.005


if verbose:
    print('###\tCreate the net')

# create the net
network = net.Network(input_size, output_size, gpu)
if gpu:
    network.cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(network.parameters(), lr)

if verbose:
    print('###\tStart training')
    start = timer()

losses = np.zeros(n_epochs)
hid = network.initHidden()
hidden = torch.rand(1, 50)

for epoch in range(n_epochs):
    input = []
    target = []
    for i in range(len(train[epoch].items)):
        input.append(train[epoch].items[i][0])
        input.append(train[epoch].items[i][1])
    #inputs = Variable(torch.tensor(inputs))
    if gpu:
        inputs = torch.cuda.FloatTensor([input])
    else:
        inputs = torch.Tensor([input])
    #for iter in range(n_iters):
    target = [float(train[epoch].z_opt/train[epoch].ub_dan), float(train[epoch].z_heu/train[epoch].ub_dan), float(train[epoch].ub_dan/train[epoch].ub_dan)]
    if gpu:
        targets = torch.cuda.FloatTensor([target])
    else:
        targets = torch.Tensor([target])
    outputs, hidden = network(inputs, hidden)
    optimizer.zero_grad()
    loss = criterion(outputs, targets)
    loss.backward(retain_graph=True)
    optimizer.step()
    losses[epoch] += loss.item()

if verbose:
    end = timer()
    print('###\tTime for training net: %.2f sec' % (end - start))


save_model(network, modelf)


write_all(losses, '', file, 'loss', '')

if verbose:
    print('###\tStart validation')
    start = timer()

network.eval()
n_epochs = int(len(validation))
error = [[0, 0, 0], [0, 0, 0]]
accuracy = [0, 0, 0]
n_iters = validation[0].n_item
for epoch in range(n_epochs):
    input = []
    targets = []
    for i in range(len(validation[epoch].items)):
        input.append(validation[epoch].items[i][0])
        input.append(validation[epoch].items[i][1])
    #inputs = Variable(torch.tensor(inputs))
    if gpu:
        inputs = torch.cuda.FloatTensor([input])
    else:
        inputs = torch.Tensor([input])
    #for iter in range(n_iters):
    target = [float(validation[epoch].z_opt/validation[epoch].ub_dan), float(validation[epoch].z_heu/validation[epoch].ub_dan), float(validation[epoch].ub_dan/validation[epoch].ub_dan)]
    if gpu:
        targets = torch.cuda.FloatTensor([target])
    else:
        targets = torch.Tensor([target])
    outputs, hidden = network(inputs, hidden)
    for i in range(len(outputs[0])):
        if (outputs[0][i]-targets[0][i])/targets[0][i] < error[0][i]:
            error[0][i] = float((outputs[0][i]-targets[0][i])/targets[0][i])
        elif (outputs[0][i]-targets[0][i])/targets[0][i] > error[1][i]:
            error[1][i] = float((outputs[0][i]-targets[0][i])/targets[0][i])
        if outputs[0][i] == targets[0][i]:
            accuracy[i] = accuracy[i] + 1
for i in range(len(accuracy)):
    accuracy[i] = accuracy[i]/(len(validation))

if verbose:
    end = timer()
    print('###\tTime for validation net: %.2f sec' % (end - start))

write_all(accuracy, '', file, 'accuracy', 'val')
write_all(error, '', file, 'error', 'val')


if verbose:
    print('###\tStart testing')
    start = timer()

network.eval()
n_epochs = int(len(test))
error = [[0, 0, 0], [0, 0, 0]]
accuracy = [0, 0, 0]
n_iters = test[0].n_item
for epoch in range(n_epochs):
    input = []
    targets = []
    for i in range(len(test[epoch].items)):
        input.append(test[epoch].items[i][0])
        input.append(test[epoch].items[i][1])
    #inputs = Variable(torch.tensor(inputs))
    if gpu:
        inputs = torch.cuda.FloatTensor([input])
    else:
        inputs = torch.Tensor([input])
    #for iter in range(n_iters):
    target = [float(test[epoch].z_opt/test[epoch].ub_dan), float(test[epoch].z_heu/test[epoch].ub_dan), float(test[epoch].ub_dan/test[epoch].ub_dan)]
    if gpu:
        targets = torch.cuda.FloatTensor([target])
    else:
        targets = torch.Tensor([target])
    outputs, hidden = network(inputs, hidden)
    for i in range(len(outputs[0])):
        if (outputs[0][i]-targets[0][i])/targets[0][i] < error[0][i]:
            error[0][i] = float((outputs[0][i]-targets[0][i])/targets[0][i])
        elif (outputs[0][i]-targets[0][i])/targets[0][i] > error[1][i]:
            error[1][i] = float((outputs[0][i]-targets[0][i])/targets[0][i])
        if outputs[0][i] == targets[0][i]:
            accuracy[i] = accuracy[i] + 1
for i in range(len(accuracy)):
    accuracy[i] = accuracy[i]/(len(test))

if verbose:
    end = timer()
    print('###\tTime for testing net: %.2f sec' % (end - start))

write_all(accuracy, '', file, 'accuracy', 'test')
write_all(error, '', file, 'error', 'test')
