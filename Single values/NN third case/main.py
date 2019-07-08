from timeit import default_timer as timer
import net as net
import utils as utils
import numpy as np
import torch.nn as nn
import torch.optim as optim 
from torch.autograd import Variable
import torch
import smtplib
import math


#############################################################################################################
#
#   DEFINE CONSTANTS
#
#############################################################################################################


gpu = True
training = True
val = True
testing = True
load_model = False
verbose = True
greedy = True
heu_base = True
plot = False
check_heu = True


path = './sorted/new_big/'
path_model = './plot/'
main = 'test_exec2'
sol = main + '_sol.txt'
file = main + '.txt'
model_f = main + '.pt'


train, validation, test, path = utils.main(verbose, path)
n_epoch = [len(train), len(validation), len(test)]

if gpu:
    torch.cuda.init()


#############################################################################################################
#
#   TRAINING
#
#############################################################################################################


n_epochs = n_epoch[0]
input_size = 400
output_size = 1  
lr = 0.005
cycle = 2

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

    # assembly input for training net
    for epoch in range(n_epochs):
        input = []
        targets = []
        for i in range(int(input_size/2)):
                if i < train[epoch].n_item:
                    input.append(train[epoch].items[i][0] / train[epoch].capacity)
                    input.append(train[epoch].items[i][1] / train[epoch].ub_dan)
                else:
                    input.append(0)
                    input.append(0)
        if gpu:
            inputs = torch.cuda.FloatTensor([input])
        else:
            inputs = torch.Tensor([input])
        target = [float(train[epoch].z_opt / train[epoch].ub_dan)]
        if gpu:
            targets = torch.cuda.FloatTensor([target])
        else:
            targets = torch.Tensor([target])

        outputs = network(inputs)  # for NN
        # update weights
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
    # save model
    utils.save_model(network, model_f)

    if verbose:
        end = timer()
        print('###\tTime for save model: %.2f sec' % (end - start))

    utils.write_all(losses, '', file, 'loss', '')

elif load_model:
    if verbose:
        print('###\tStart loading model')
        start = timer()

    # load model
    network = utils.load_model(path_model + model_f, input_size, output_size, gpu)
    if gpu:
        network.cuda()

    if verbose:
        end = timer()
        print('###\tTime for load model: %.2f sec' % (end - start))


#############################################################################################################
#
#   VALIDATION
#
#############################################################################################################


if val:
    if verbose:
        print('###\tStart validation')
        start = timer()

    network.eval()
    n_epochs = n_epoch[1]
    error = [0, 0, 0]
    accuracy = 0
    # assembly input for validate net
    for epoch in range(n_epochs):
        input = []
        targets = []
        for i in range(int(input_size / 2)):
                if i < validation[epoch].n_item:
                    input.append(validation[epoch].items[i][0] / validation[epoch].capacity)
                    input.append(validation[epoch].items[i][1] / validation[epoch].ub_dan)
                else:
                    input.append(0)
                    input.append(0)
        if gpu:
            inputs = torch.cuda.FloatTensor([input])
        else:
            inputs = torch.Tensor([input])
        target = [float(validation[epoch].z_opt / validation[epoch].ub_dan)]
        if gpu:
            targets = torch.cuda.FloatTensor([target])
        else:
            targets = torch.Tensor([target])
        # prediction
        outputs = network(inputs)  # for NN
        outputs = float(outputs) * validation[epoch].ub_dan
        targets = float(targets) * validation[epoch].ub_dan
        # compute error and accuracy
        if (outputs - targets) / targets < error[0]:
            error[0] = float((outputs - targets) / targets)
        elif (outputs - targets) / targets > error[1]:
            error[1] = float((outputs - targets) / targets)
        error[2] = error[2] + abs(float((outputs - targets) / targets))
        if outputs == targets:
            accuracy = accuracy + 1
    accuracy = accuracy / n_epochs
    error[2] = error[2] / n_epochs

    if verbose:
        end = timer()
        print('###\tTime for validation net: %.2f sec' % (end - start))

    utils.write_all(accuracy, '', file, 'accuracy', 'val')
    utils.write_all(error, '', file, 'error', 'val')


#############################################################################################################
#
#   TESTING
#
#############################################################################################################


if testing:
    if verbose:
        print('###\tStart testing')
        start = timer()

    network.eval()
    n_epochs = n_epoch[2]
    error = [0, 0, 0]
    accuracy = 0
    # assembly input for net
    for epoch in range(n_epochs):
        input = []
        targets = []
        for i in range(int(input_size / 2)):
                if i < test[epoch].n_item:
                    input.append(test[epoch].items[i][0] / test[epoch].capacity)
                    input.append(test[epoch].items[i][1] / test[epoch].ub_dan)
                else:
                    input.append(0)
                    input.append(0)
        if gpu:
            inputs = torch.cuda.FloatTensor([input])
        else:
            inputs = torch.Tensor([input])
        target = [float(test[epoch].z_opt / test[epoch].ub_dan)]
        if gpu:
            targets = torch.cuda.FloatTensor([target])
        else:
            targets = torch.Tensor([target])
        # prediction
        outputs = network(inputs)  # for NN
        outputs = float(outputs) * test[epoch].ub_dan
        targets = float(targets) * test[epoch].ub_dan
        # compute error and accuracy
        if (outputs - targets) / targets < error[0]:
            error[0] = float((outputs - targets) / targets)
        elif (outputs - targets) / targets > error[1]:
            error[1] = float((outputs - targets) / targets)
        error[2] = error[2] + abs(float((outputs - targets) / targets))
        if outputs == targets:
            accuracy = accuracy + 1
    accuracy = accuracy / n_epochs
    error[2] = error[2] / n_epochs

    if verbose:
        end = timer()
        print('###\tTime for testing net: %.2f sec' % (end - start))

    utils.write_all(accuracy, '', file, 'accuracy', 'test')
    utils.write_all(error, '', file, 'error', 'test')
