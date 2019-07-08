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
    path = './sorted/'
    f = 'param1.txt'
    m = 'param1.pt'
    params = sys.argv
    if len(params) > 1:
        for i in range(len(params)):
            if params[i] == '--path':
                if len(params) > i+1:
                    path = params[i+1]
                else:
                    usage(True)
                    sys.exit()
            elif params[i] == '--v':
                verbose = True
            elif params[i] == '--help' or params[i] == '-h':
                usage(False)
                sys.exit()
            else:
                file = params[i].split('.')
                if (file[1] == 'py'):
                    name = file[0].split('/')
                    f = name[-1]+'.txt'
                    m = name[-1]+'.pt'
    return path, verbose, f, m


def main(sourcep):
    # read all arguments
    path, verbose, file, modelf = read_params(sourcep)
    if verbose:
        print('###\tStart reading data')
    start = timer()
    train, validation, test = dt.read_data(path)
    end = timer()
    if verbose:
        print('###\tTime for reading data: %.2f sec' % (end - start))
    return train, validation, test, verbose, path, file, modelf


# order to execute
torch.cuda.init()
path = './sorted/'
lossfile = 'losses.txt'
train, validation, test, verbose, path, file, modelf = main(path)
n_epochs = len(train)
n_iters = train[0].n_item
input_size = 2
h1 = 512
h2 = 256
h3 = 128
h4 = 64
h5 = 32
output_size = 3 # fixed
lr = 0.005


if verbose:
    print('###\tCreate the net')

# create the net
rnn = net.Network(input_size, n_iters, output_size, False, lr, h1, h2, h3, h4, h5)
criterion = nn.MSELoss()
optimizer = optim.SGD(rnn.parameters(), lr)


if verbose:
    print('###\tStart training')
    start = timer()

losses = np.zeros(n_epochs)

for epoch in range(n_epochs):
    inputs = []
    targets = []
    for i in range(len(train[epoch].items)):
        inputs.append(train[epoch].items[i][0])
        inputs.append(train[epoch].items[i][1])
    inputs = Variable(torch.tensor(inputs))

    #for iter in range(n_iters):
    targets = [float(train[epoch].z_opt/train[epoch].ub_dan), float(train[epoch].z_heu/train[epoch].ub_dan), float(train[epoch].ub_dan/train[epoch].ub_dan)]
    targets = Variable(torch.tensor(targets))

    outputs = rnn(inputs)
    optimizer.zero_grad()
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    losses[epoch] += loss.item()
if verbose:
    end = timer()
    print('###\tTime for training net: %.2f sec' % (end - start))

save_model(rnn, modelf)

write_all(losses, '', file, 'loss', '')

if verbose:
    print('###\tStart validation')
    start = timer()

rnn.eval()
n_epochs = len(validation)
error = [[0, 0, 0], [0, 0, 0]]
accuracy = [0, 0, 0]
n_iters = validation[0].n_item
for epoch in range(n_epochs):
    inputs = []
    targets = []
    for i in range(len(validation[epoch].items)):
        inputs.append(validation[epoch].items[i][0])
        inputs.append(validation[epoch].items[i][1])
    inputs = Variable(torch.tensor(inputs))

    targets = [float(validation[epoch].z_opt/validation[epoch].ub_dan), float(validation[epoch].z_heu/validation[epoch].ub_dan), float(validation[epoch].ub_dan/validation[epoch].ub_dan)]
    targets = Variable(torch.tensor(targets))
    outputs = rnn(inputs)
    for i in range(len(outputs)):
        if (outputs[i]-targets[i])/targets[i] < error[0][i]:
            error[0][i] = float((outputs[i]-targets[i])/targets[i])
        elif (outputs[i]-targets[i])/targets[i] > error[1][i]:
            error[1][i] = float((outputs[i]-targets[i])/targets[i])
    if outputs[0] == targets[0]:
        accuracy[0] = accuracy[0] + 1
    if outputs[1] == targets[1]:
        accuracy[1] = accuracy[1] + 1
    if outputs[2] == targets[2]:
        accuracy[2] = accuracy[2] + 1
for i in range(len(accuracy)):
    accuracy[i] = accuracy[i]/len(validation)
if verbose:
    end = timer()
    print('###\tTime for validation net: %.2f sec' % (end - start))

write_all(accuracy, '', file, 'accuracy', 'val')
write_all(error, '', file, 'error', 'val')

if verbose:
    print('###\tStart testing')
    start = timer()

rnn.eval()
n_epochs = len(test)
error = [[0, 0, 0], [0, 0, 0]]
accuracy = [0, 0, 0]
n_iters = test[0].n_item
for epoch in range(n_epochs):
    inputs = []
    targets = []
    for i in range(len(test[epoch].items)):
        inputs.append(test[epoch].items[i][0])
        inputs.append(test[epoch].items[i][1])
    inputs = Variable(torch.tensor(inputs))

    targets = [float(test[epoch].z_opt/test[epoch].ub_dan), float(test[epoch].z_heu/test[epoch].ub_dan), float(test[epoch].ub_dan/test[epoch].ub_dan)]
    targets = Variable(torch.tensor(targets))
    outputs = rnn(inputs)
    for i in range(len(outputs)):
        if (outputs[i]-targets[i])/targets[i] < error[0][i]:
            error[0][i] = float((outputs[i]-targets[i])/targets[i])
        elif (outputs[i]-targets[i])/targets[i] > error[1][i]:
            error[1][i] = float((outputs[i]-targets[i])/targets[i])
    if outputs[0] == targets[0]:
        accuracy[0] = accuracy[0] + 1
    if outputs[1] == targets[1]:
        accuracy[1] = accuracy[1] + 1
    if outputs[2] == targets[2]:
        accuracy[2] = accuracy[2] + 1
for i in range(len(accuracy)):
    accuracy[i] = accuracy[i]/len(test)
if verbose:
    end = timer()
    print('###\tTime for testing net: %.2f sec' % (end - start))

write_all(accuracy, '', file, 'accuracy', 'test')
write_all(error, '', file, 'error', 'test')
