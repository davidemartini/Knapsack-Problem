from timeit import default_timer as timer
import utils as utils
import numpy as np
import torch.nn as nn
import torch.optim as opt
import torch
import Bnet as net_opt
import Pnet as net_sol


#############################################################################################################
#
#   DEFINE CONSTANTS
#
#############################################################################################################


gpu = False

verbose = True

training = False

greedy = True

mod = 'opt'
if training:
    mod = 'solution'
    greedy = False

plot = False

path = './sorted/new_big/'
path_plot = './plot/'
path_model = './model/'
main = 'KP'
sol = main + '_sol.txt'
file = main + '.txt'
model_f = main + '.pt'
model_load = 'KP_sol.pt'
model_items = 'KP_pr.pt'
new_test = main + '_new.txt'

val = training
testing = training
load_model = not training
heu_base = greedy
heu_inc = greedy
heu_b_inc = greedy
heu_rand = greedy
heu_prob = greedy
inv_greedy = greedy
check_heu = greedy


train, validation, test, path = utils.main(verbose, path)
n_epoch = [len(train), len(validation), len(test)]

if gpu:
    torch.cuda.init()

if mod == 'opt':
    utils.initialize(path_plot, file)
    utils.initialize(path_plot, sol)
elif mod == 'solution' and training:
    utils.initialize(path_plot, new_test)
else:
    utils.initialize(path_plot, file)
    utils.initialize(path_plot, sol)

start = timer()


#############################################################################################################
#
#   TRAINING
#
#############################################################################################################


n_epochs = n_epoch[0]
input_size = 400
output_size = 1
cycle = 1
lr = 0.005

if mod == 'solution':
    output_size = 200
    cycle = 5
    lr = 0.001

losses = []

if training:
    if verbose:
        print('###\tCreate the net')

        # create the net
        network = net_sol.Network(input_size, output_size, gpu)
        if network.gpu:
            network.cuda()

        criterion = nn.MSELoss()
        optimizer = opt.SGD(network.parameters(), lr)

        if verbose:
            print('###\tStart training')
            start = timer()

        losses = np.zeros(n_epochs)
        hid = network.initHidden()

        for cl in range(cycle):
            # assembly input for training net
            for epoch in range(n_epochs):
                input_t = []
                target = []
                for i in range(int(input_size/2)):
                        if i < train[epoch].n_item:
                            input_t.append(train[epoch].items[i][0] / train[epoch].capacity)
                            input_t.append(train[epoch].items[i][1] / train[epoch].ub_dan)
                        else:
                            input_t.append(0)
                            input_t.append(0)
                if network.gpu:
                    inputs = torch.cuda.FloatTensor([input_t])
                else:
                    inputs = torch.Tensor([input_t])

                if mod == 'opt':
                    target = [float(train[epoch].z_opt / train[epoch].ub_dan)]
                elif mod == 'solution':
                    for i in range(int(input_size / 2)):
                        target.append(0)
                    for j in range(len(train[epoch].solutions)):
                        target[train[epoch].solutions[j]] = 1

                if network.gpu:
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

    utils.write_all(losses, '', file, 'loss', '', False, -1, -1)

elif load_model:
    if verbose:
        print('###\tStart loading model')
        start = timer()

    # load model
    network_opt = utils.load_model(path_model + model_load, input_size, 1, gpu, 'opt')

    network_sol = utils.load_model(path_model + model_items, input_size, 200, gpu, 'solution')

    if network_opt.gpu:
        network_opt.cuda()
    if network_sol.gpu:
        network_sol.cuda()

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
    error = []
    accuracy = 0
    tot_card = 0
    if mod == 'opt':
        error = [0, 0, 0]
    elif mod == 'solution':
        error = 0
    # assembly input for validate net
    for epoch in range(n_epochs):
        input_t = []
        target = []
        for i in range(int(input_size / 2)):
                if i < validation[epoch].n_item:
                    input_t.append(validation[epoch].items[i][0] / validation[epoch].capacity)
                    input_t.append(validation[epoch].items[i][1] / validation[epoch].ub_dan)
                else:
                    input_t.append(0)
                    input_t.append(0)
        if network.gpu:
            inputs = torch.cuda.FloatTensor([input_t])
        else:
            inputs = torch.Tensor([input_t])

        if mod == 'opt':
            target = [float(validation[epoch].z_opt / validation[epoch].ub_dan)]
        elif mod == 'solution':
            for i in range(int(input_size / 2)):
                target.append(0)
            for j in range(len(validation[epoch].solutions)):
                target[validation[epoch].solutions[j]] = 1

        if network.gpu:
            targets = torch.cuda.FloatTensor([target])
        else:
            targets = torch.Tensor([target])

        # prediction
        outputs = network(inputs)  # for NN
        if mod == 'opt':
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
        elif mod == 'solution':
            # compute error and accuracy
            out = []
            for i in range(len(outputs[0])):
                if float(outputs[0][i])*10 > 0.7:
                    out.append(1)
            # compute error and accuracy
            if utils.compute_feasibility(validation, epoch, out) == -1:
                # compute greedy solution
                z_out, out_sol = utils.heuristic_b_inc_pr(validation, epoch, input_size, network, out)
            else:
                z_out = utils.compute_z(validation, out, epoch)
            z = utils.compute_z(validation, validation[epoch].solutions, epoch)
            error += abs(z - z_out) / z

    if mod == 'opt':
        accuracy = accuracy / n_epochs
        error[2] = error[2] / n_epochs
    elif mod == 'solution':
        f_out = open(path_plot + new_test, 'a')
        f_out.write('ERROR VALIDATION\n')
        f_out.write(str(error / n_epochs))
        f_out.write('\n')
        f_out.close()

    if verbose:
        end = timer()
        print('###\tTime for validation net: %.2f sec' % (end - start))

    if mod == 'opt':
        utils.write_all(accuracy, '', file, 'accuracy', 'val', False, -1, -1)
        utils.write_all(error, '', file, 'error', 'val', False, -1, -1)


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
    tot_card = 0
    error = [0, 0, 0]
    accuracy = 0
    pos = 0
    neg = 0
    if mod == 'opt':
        error = [0, 0, 0]
    elif mod == 'solution':
        error = [0, [0, -1], [1, 0]]

    # assembly input for net
    for epoch in range(n_epochs):
        input_t = []
        target = []
        for i in range(int(input_size / 2)):
                if i < test[epoch].n_item:
                    input_t.append(test[epoch].items[i][0] / test[epoch].capacity)
                    input_t.append(test[epoch].items[i][1] / test[epoch].ub_dan)
                else:
                    input_t.append(0)
                    input_t.append(0)
        if network.gpu:
            inputs = torch.cuda.FloatTensor([input_t])
        else:
            inputs = torch.Tensor([input_t])

        if mod == 'opt':
            target = [float(test[epoch].z_opt / test[epoch].ub_dan)]
        elif mod == 'solution':
            for i in range(int(input_size / 2)):
                target.append(0)
            for j in range(len(test[epoch].solutions)):
                target[test[epoch].solutions[j]] = 1

        if network.gpu:
            targets = torch.cuda.FloatTensor([target])
        else:
            targets = torch.Tensor([target])

        # prediction
        outputs = network(inputs)  # for NN

        if mod == 'opt':
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
        elif mod == 'solution':
            out = []
            max = 0
            for i in range(len(outputs[0])):
                if outputs[0][i] > max:
                    max = outputs[0][i]

            for i in range(len(outputs[0])):
                outputs[0][i] = outputs[0][i] / max
                if float(outputs[0][i])*10 > 0.5:
                    out.append(1)
            # compute error and accuracy
            if utils.compute_feasibility(test, epoch, out) == -1:
                # compute greedy solution
                z_out, out_sol = utils.heuristic_b_inc_pr(test, epoch, input_size, network, out)
            else:
                z_out = utils.compute_z(test, out, epoch)
            z = utils.compute_z(test, test[epoch].solutions, epoch)
            error[0] += abs(z - z_out) / z
            if (z_out - z) / z < 0:
                neg += 1
                if error[1][0] > (z_out - z) / z:
                    error[1][0] = (z_out - z) / z
                if error[1][1] < (z_out - z) / z:
                    error[1][1] = (z_out - z) / z
            if (z_out - z) / z > 0:
                pos += 1
                if error[2][0] > (z_out - z) / z:
                    error[2][0] = (z_out - z) / z
                if error[2][1] < (z_out - z) / z:
                    error[2][1] = (z_out - z) / z

    if mod == 'opt':
        accuracy = accuracy / n_epochs
        error[2] = error[2] / n_epochs
    elif mod == 'solution':
        f_out = open(path_plot + new_test, 'a')
        for i in range(len(error)):
            if i == 0:
                f_out.write('ERROR TESTING\n')
                f_out.write(str(error[i] / n_epochs))
            elif i == 1:
                f_out.write('MIN NEG\n')
                f_out.write(str(error[i][0]))
                f_out.write('\nMAX NEG\n')
                f_out.write(str(error[i][1]))
                f_out.write('\nNUM NEG\n')
                f_out.write(str(neg))
            else:
                f_out.write('MIN POS\n')
                f_out.write(str(error[i][0]))
                f_out.write('\nMAX POS\n')
                f_out.write(str(error[i][1]))
                f_out.write('\nNUM POS\n')
                f_out.write(str(pos))
            f_out.write('\n')

        f_out.close()

    if verbose:
        end = timer()
        print('###\tTime for testing net: %.2f sec' % (end - start))

    if mod == 'opt':
        utils.write_all(accuracy, '', file, 'accuracy', 'test', False, -1, -1)
        utils.write_all(error, '', file, 'error', 'test', False, -1, -1)


#############################################################################################################
#
#   GREEDY SOLUTIONS
#
#############################################################################################################


if greedy:
    if verbose:
        print('###\tStart compute greedy solutions')
        start = timer()

    n_epochs = n_epoch[2]
    data = []
    for i in range(n_epochs):
        # compute greedy solution
        opt_sol = utils.greedy(test, i)
        data.append(opt_sol)

    utils.write_sol(data, 0, path_plot, sol, '')

    if verbose:
        end = timer()
        print('###\tTime for compute greedy solutions: %.2f sec' % (end - start))


#############################################################################################################
#
#   INVERSE GREEDY
#
#############################################################################################################


if inv_greedy and mod == 'opt':
    if verbose:
        print('###\tStart inverse greedy')
        start = timer()

    n_epochs = n_epoch[2]
    data = []

    # base estimation single case
    for i in range(n_epochs):
        # compute greedy solution
        opt_sol = utils.inv_greedy(test, i)
        data.append(opt_sol)

    utils.write_sol(data, 1, path_plot, sol, 'inv_greedy')

    if verbose:
        end = timer()
        print('###\tTime for inverse greedy: %.2f sec' % (end - start))


#############################################################################################################
#
#   HEURISTIC BASE
#
#############################################################################################################


if heu_base:
    if verbose:
        print('###\tStart heuristic base')
        start = timer()

    network_opt.eval()
    n_epochs = n_epoch[2]
    data = []

    # base estimation single case
    for i in range(n_epochs):
        opt_sol, _ = utils.heuristic_base(test, i, input_size, network_opt)
        data.append(opt_sol)

    utils.write_sol(data, 1, path_plot, sol, 'base')

    if verbose:
        end = timer()
        print('###\tTime for heuristic base: %.2f sec' % (end - start))


#############################################################################################################
#
#   HEURISTIC WITH INCUMBENT
#
#############################################################################################################


if heu_inc:
    if verbose:
        print('###\tStart heuristic with incumbent')
        start = timer()

    network_opt.eval()
    n_epochs = n_epoch[2]
    error = [0, 0, 0]
    accuracy = 0
    data = []

    # base estimation single case
    for epoch in range(n_epochs):
        # compute solution
        z_sol, opt_sol = utils.heuristic_inc(test, epoch, input_size, network_opt, [])
        # compute greedy solution
        data.append(opt_sol)

    utils.write_sol(data, 1, path_plot, sol, 'incumbent')

    if verbose:
        end = timer()
        print('###\tTime for heuristic with incumbent: %.2f sec' % (end - start))


#############################################################################################################
#
#   HEURISTIC WITH BEST INCUMBENT
#
#############################################################################################################


if heu_b_inc:
    if verbose:
        print('###\tStart heuristic with best incumbent')
        start = timer()

    network_opt.eval()
    n_epochs = n_epoch[2]
    error = [0, 0, 0]
    accuracy = 0
    data = []

    # base estimation single case
    for epoch in range(n_epochs):
        # compute solution
        z_sol, opt_sol = utils.heuristic_b_inc(test, epoch, input_size, network_opt)
        # compute greedy solution
        data.append(opt_sol)

    utils.write_sol(data, 1, path_plot, sol, 'best_incumbent')

    if verbose:
        end = timer()
        print('###\tTime for heuristic with best incumbent: %.2f sec' % (end - start))


#############################################################################################################
#
#   HEURISTIC WITH BEST INCUMBENT --- RANDOM ITEMS ---
#
#############################################################################################################


if heu_rand:
    if verbose:
        print('###\tStart heuristic with best incumbent --- random items ---')
        start = timer()

    network_opt.eval()
    n_epochs = n_epoch[2]

    data = []

    # base estimation single case
    for epoch in range(n_epochs):
        # compute solution
        z_sol, opt_sol = utils.heuristic_rand(test, epoch, input_size, network_opt)
        # compute greedy solution
        data.append(opt_sol)

    utils.write_sol(data, 1, path_plot, sol, 'heu_rand')

    if verbose:
        end = timer()
        print('###\tTime for heuristic with best --- random items ---: %.2f sec' % (end - start))


#############################################################################################################
#
#   HEURISTIC WITH BEST INCUMBENT --- PROBABILITY ITEMS ---
#
#############################################################################################################


if heu_prob:
    if verbose:
        print('###\tStart heuristic with best incumbent --- probability items ---')
        start = timer()

    network_sol.eval()
    n_epochs = n_epoch[2]
    data = []

    # assembly input for net
    for epoch in range(n_epochs):
        input_t = []
        target = []
        for i in range(int(input_size / 2)):
                if i < test[epoch].n_item:
                    input_t.append(test[epoch].items[i][0] / test[epoch].capacity)
                    input_t.append(test[epoch].items[i][1] / test[epoch].ub_dan)
                else:
                    input_t.append(0)
                    input_t.append(0)
        if network_sol.gpu:
            inputs = torch.cuda.FloatTensor([input_t])
        else:
            inputs = torch.Tensor([input_t])

            for i in range(int(input_size / 2)):
                target.append(0)
            for j in range(len(test[epoch].solutions)):
                target[test[epoch].solutions[j]] = 1

        if network_sol.gpu:
            targets = torch.cuda.FloatTensor([target])
        else:
            targets = torch.Tensor([target])

        # prediction
        outputs = network_sol(inputs)  # for NN

        out = []
        for i in range(len(outputs[0])):
            if float(outputs[0][i])*10 > 0.7:
                out.append(1)

        # compute error and accuracy
        if utils.compute_feasibility(test, epoch, out) == -1:

            # compute greedy solution
            z_out, out_sol = utils.heuristic_b_inc_pr(test, epoch, input_size, network_sol, out)
            data.append(out_sol)
        else:
            z_out = utils.compute_z(test, out, epoch)
            data.append(out)

    utils.write_sol(data, 1, path_plot, sol, 'heu_prob')

    if verbose:
        end = timer()
        print('###\tTime for heuristic with best incumbent --- probability items ---: %.2f sec' % (end - start))


#############################################################################################################
#
#   CHECK HEURISTIC SOLUTIONS
#
#############################################################################################################


if check_heu:
    if verbose:
        print('###\tStart read heuristic solutions')
        start = timer()

    # read solution
    greedy, heuristic_base, heuristic_inc, heuristic_b_inc, inv_greedy, heuristic_rand, heuristic_prob = utils.read_sol(path_plot, sol, n_epoch[2])
    for i in range(len(heuristic_base)):
        w = 0
        for j in range(len(heuristic_base[i])):
            w += test[i].items[heuristic_base[i][j]][0]

    # check solution
    if utils.check_heu_sol(heuristic_base, test) == -1:
        print('WRONG HEURISTIC BASE SOLUTION')
    elif utils.check_heu_sol(heuristic_inc, test) == -1:
        print('WRONG HEURISTIC INCUMBENT SOLUTION')
    elif utils.check_heu_sol(heuristic_inc, test) == -1:
        print('WRONG HEURISTIC BEST INCUMBENT SOLUTION')
    elif utils.check_heu_sol(heuristic_rand, test) == -1:
        print('WRONG HEURISTIC BEST INCUMBENT SOLUTION --- RANDOM ITEMS ---')
    elif utils.check_heu_sol(heuristic_prob, test) == -1:
        print('WRONG HEURISTIC BEST INCUMBENT SOLUTION --- PROBABILITY ITEMS ---')
    elif utils.check_heu_sol(heuristic_inc, test) == -1:
        print('WRONG INVERSE GREEDY SOLUTION')
    else:
        # heuristic performance

        accuracy, error, better = utils.heuristic_performance(test, 0, inv_greedy, n_epoch[2], 'solver')
        utils.write_all(accuracy, path_plot, file, 'accuracy', 'inv_solver', False, -1, -1)
        utils.write_all(error, path_plot, file, 'error', 'inv_solver', True, better[2], better[3])

        accuracy, error, better = utils.heuristic_performance(test, greedy, inv_greedy, n_epoch[2], 'greedy')
        utils.write_all(accuracy, path_plot, file, 'accuracy', 'inv_greedy', False, -1, -1)
        utils.write_all(error, path_plot, file, 'error', 'inv_greedy', True, better[2], better[3])

        accuracy, error, better = utils.heuristic_performance(test, 0, heuristic_base, n_epoch[2], 'solver')
        utils.write_all(accuracy, path_plot, file, 'accuracy', 'heu_base_solver', False, -1, -1)
        utils.write_all(error, path_plot, file, 'error', 'heu_base_solver', True, better[2], better[3])

        accuracy, error, better = utils.heuristic_performance(test, greedy, heuristic_base, n_epoch[2], 'greedy')
        utils.write_all(accuracy, path_plot, file, 'accuracy', 'heu_base_greedy', False, -1, -1)
        utils.write_all(error, path_plot, file, 'error', 'heu_base_greedy', True, better[2], better[3])

        accuracy, error, better = utils.heuristic_performance(test, 0, heuristic_inc, n_epoch[2], 'solver')
        utils.write_all(accuracy, path_plot, file, 'accuracy', 'heu_inc_solver', False, -1, -1)
        utils.write_all(error, path_plot, file, 'error', 'heu_inc_solver', True, better[2], better[3])

        accuracy, error, better = utils.heuristic_performance(test, greedy, heuristic_inc, n_epoch[2], 'greedy')
        utils.write_all(accuracy, path_plot, file, 'accuracy', 'heu_inc_greedy', False, -1, -1)
        utils.write_all(error, path_plot, file, 'error', 'heu_inc_greedy', True, better[2], better[3])

        accuracy, error, better = utils.heuristic_performance(test, 0, heuristic_b_inc, n_epoch[2], 'solver')
        utils.write_all(accuracy, path_plot, file, 'accuracy', 'heu_b_inc_solver', False, -1, -1)
        utils.write_all(error, path_plot, file, 'error', 'heu_b_inc_solver', True, better[2], better[3])

        accuracy, error, better = utils.heuristic_performance(test, greedy, heuristic_b_inc, n_epoch[2], 'greedy')
        utils.write_all(accuracy, path_plot, file, 'accuracy', 'heu_b_inc_greedy', False, -1, -1)
        utils.write_all(error, path_plot, file, 'error', 'heu_b_inc_greedy', True, better[2], better[3])

        accuracy, error, better = utils.heuristic_performance(test, 0, heuristic_rand, n_epoch[2], 'solver')
        utils.write_all(accuracy, path_plot, file, 'accuracy', 'heu_rand_solver', False, -1, -1)
        utils.write_all(error, path_plot, file, 'error', 'heu_rand_solver', True, better[2], better[3])

        accuracy, error, better = utils.heuristic_performance(test, greedy, heuristic_rand, n_epoch[2], 'greedy')
        utils.write_all(accuracy, path_plot, file, 'accuracy', 'heu_rand_greedy', False, -1, -1)
        utils.write_all(error, path_plot, file, 'error', 'heu_rand_greedy', True, better[2], better[3])

        accuracy, error, better = utils.heuristic_performance(test, 0, heuristic_prob, n_epoch[2], 'solver')
        utils.write_all(accuracy, path_plot, file, 'accuracy', 'heu_prob_solver', False, -1, -1)
        utils.write_all(error, path_plot, file, 'error', 'heu_prob_solver', True, better[2], better[3])

        accuracy, error, better = utils.heuristic_performance(test, greedy, heuristic_prob, n_epoch[2], 'greedy')
        utils.write_all(accuracy, path_plot, file, 'accuracy', 'heu_prob_greedy', False, -1, -1)
        utils.write_all(error, path_plot, file, 'error', 'heu_prob_greedy', True, better[2], better[3])

    if verbose:
        end = timer()
        print('###\tTime for check heuristic solutions: %.2f sec' % (end - start))
