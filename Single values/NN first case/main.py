import sys
import os
import data as dt
from timeit import default_timer as timer
import net as net
import numpy as np
import torch.nn as nn
import torch.optim as optim 
from torch.autograd import Variable
import torch
import smtplib
import math


gpu = True
if gpu:
    torch.cuda.init()


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
        fout.write(str(data)+" ")
        fout.write('\n')
        fout.close()
    elif measure == 'error':
        fout = open('plot/'+file, 'a')
        if type == 'val':
            fout.write('ERROR VALIDATION\n')
        elif type == 'test':
            fout.write('ERROR TEST\n')
        for j in range(len(data)):
            fout.write(str(data[j])+" ")
            fout.write('\n')
        fout.write('\n')
        fout.close()


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path_model, input_size, output_size, gpu):
    model = net.Network(input_size, output_size, gpu)
    model.load_state_dict(torch.load(path_model))
    return model


def read_params(path):
    verbose = True
    gpu = True
    path = './sorted/unfix/'
    f = 't0Bnet.txt'
    m = 't0Bnet.pt'
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
    if file[1] == 'py':
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
path = './sorted/unfix/'
path_model = './plot/'
train, validation, test, verbose, path, file, modelf, gpu = main(path)
gpu = True
training = True
n_epochs = len(train)
n_iters = train[0].n_item
input_size = 400
output_size = 1 
lr = 0.005
cycle = 5

if training:
    if verbose:
        print('###\tCreate the net')

    # create the net
    network = net.Network(input_size, output_size, gpu)
    if gpu:
        network.cuda()
    criterion = nn.L1Loss()
    optimizer = optim.SGD(network.parameters(), lr)

    if verbose:
        print('###\tStart training')
        start = timer()

    losses = np.zeros(n_epochs)
    hid = network.initHidden()
    for k in range(cycle):
        for epoch in range(n_epochs):
            input = []
            target = []
            for i in range(int(input_size/2)):
                if i < train[epoch].n_item:
                    input.append(train[epoch].items[i][0])
                    input.append(train[epoch].items[i][1])
                else:
                    input.append(2)
                    input.append(-1)
            if gpu:
                inputs = torch.cuda.FloatTensor([input])
            else:
                inputs = torch.Tensor([input])
            target = [float(train[epoch].z_opt/train[epoch].ub_dan)]
            if gpu:
                targets = torch.cuda.FloatTensor([target])
            else:
                targets = torch.Tensor([target])
            outputs = network(inputs)  # for NN
            optimizer.zero_grad()
            loss = criterion(outputs, targets)

            loss.backward(retain_graph=True)
            optimizer.step()
            losses[epoch] += loss.item()

    if verbose:
        end = timer()
        print('###\tTime for training net: %.2f sec' % (end - start))

    if verbose:
        print('###\tStart saving model')
        start = timer()

    save_model(network, modelf)

    if verbose:
        end = timer()
        print('###\tTime for save model: %.2f sec' % (end - start))

    write_all(losses, '', file, 'loss', '')
else:
    if verbose:
        print('###\tStart loading model')
        start = timer()

    network = load_model(path_model+modelf, input_size, output_size, gpu)
    if gpu:
        network.cuda()

    if verbose:
        end = timer()
        print('###\tTime for load model: %.2f sec' % (end - start))


if verbose:
    print('###\tStart validation')
    start = timer()

network.eval()
n_epochs = len(validation)
error = [0, 0, 0]
accuracy = 0
n_iters = validation[0].n_item
for epoch in range(n_epochs):
    input = []
    targets = []
    for i in range(int(input_size/2)):
            if i < validation[epoch].n_item:
                input.append(validation[epoch].items[i][0])
                input.append(validation[epoch].items[i][1])
            else:
                input.append(0)
                input.append(0)
    if gpu:
        inputs = torch.cuda.FloatTensor([input])
    else:
        inputs = torch.Tensor([input])
    target = [float(validation[epoch].z_opt/validation[epoch].ub_dan)]
    if gpu:
        targets = torch.cuda.FloatTensor([target])
    else:
        targets = torch.Tensor([target])
    outputs = network(inputs)  # for NN
    if (outputs[0]-targets[0])/targets[0] < error[0]:
        error[0] = float((outputs[0]-targets[0])/targets[0])
    elif (outputs[0]-targets[0])/targets[0] > error[1]:
        error[1] = float((outputs[0]-targets[0])/targets[0])
    error[2] = error[2] + abs(float((outputs[0]-targets[0])/targets[0]))
    if outputs[0] == targets[0]:
        accuracy = accuracy + 1
accuracy = accuracy/len(validation)
error[2] = error[2]/len(validation)
if verbose:
    end = timer()
    print('###\tTime for validation net: %.2f sec' % (end - start))

write_all(accuracy, '', file, 'accuracy', 'val')
write_all(error, '', file, 'error', 'val')

if verbose:
    print('###\tStart testing')
    start = timer()

network.eval()
n_epochs = len(test)
error = [0, 0, 0]
accuracy = 0
n_iters = test[0].n_item
for epoch in range(n_epochs):
    input = []
    targets = []
    for i in range(int(input_size/2)):
            if i < test[epoch].n_item:
                input.append(test[epoch].items[i][0])
                input.append(test[epoch].items[i][1])
            else:
                input.append(2)
                input.append(-1)
    if gpu:
        inputs = torch.cuda.FloatTensor([input])
    else:
        inputs = torch.Tensor([input])
    target = [float(test[epoch].z_opt/test[epoch].ub_dan)]
    if gpu:
        targets = torch.cuda.FloatTensor([target])
    else:
        targets = torch.Tensor([target])
    outputs = network(inputs)  # for NN
    if (outputs[0]-targets[0])/targets[0] < error[0]:
        error[0] = float((outputs[0]-targets[0])/targets[0])
    elif (outputs[0]-targets[0])/targets[0] > error[1]:
        error[1] = float((outputs[0]-targets[0])/targets[0])
    error[2] = error[2] + abs(float((outputs[0]-targets[0])/targets[0]))
    if outputs[0] == targets[0]:
        accuracy = accuracy + 1
accuracy = accuracy/len(test)
error[2] = error[2]/len(test)

if verbose:
    end = timer()
    print('###\tTime for testing net: %.2f sec' % (end - start))

write_all(accuracy, '', file, 'accuracy', 'test')
write_all(error, '', file, 'error', 'test')

