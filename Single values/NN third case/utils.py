import sys
from timeit import default_timer as timer
import data as dt
import os
import torch


def usage(err):
    if err:
        print('Error in --path. Use:')
    print('<program.py> --help | --path path | --v')


def write_sol(data, type, path, file):
    if path == '':
        print('Wrong path for write sol type %d' % type)
    else:
        # GREEDY
        if type == 0:
            fout = open(path + file, 'w')
            fout.write('GREEDY SOLUTION\n')
            for j in range(len(data)):
                for i in range(len(data[j])):
                    if i != len(data[j]) - 1:
                        fout.write(str(data[j][i] + 1) + ' ')
                    else:
                        fout.write(str(data[j][i] + 1))
                fout.write('\n')
        # HEURISTIC BASE
        elif type == 1:
            fout = open(path + file, 'a')
            fout.write('HEURISTIC BASE SOLUTION\n')
            for j in range(len(data)):
                for i in range(len(data[j])):
                    if i != len(data[j]) - 1:
                        fout.write(str(data[j][i] + 1) + ' ')
                    else:
                        fout.write(str(data[j][i] + 1))
                fout.write('\n')
        fout.close()


def read_sol(path, file, n):
    greedy = []
    heuristic_base = []
    lines = open(path + file, 'r').read().split('\n')
    for i in range(len(lines)):
        if lines[i] == 'GREEDY SOLUTION':
            for j in range(i + 1, i + 1 + n):
                line = lines[j].split(' ')
                tmp = []
                for k in range(len(line)):
                    tmp.append((int(line[k])))
                greedy.append(tmp)
            i = i + n
        elif lines[i] == 'HEURISTIC BASE SOLUTION':
            for j in range(i + 1, i + 1 + n):
                line = lines[j].split(' ')
                tmp = []
                for k in range(len(line)):
                    tmp.append((int(line[k])))
                heuristic_base.append(tmp)
            i = i + n
    return greedy, heuristic_base


def write_all(data, path, file, measure, type):
    if path != '':
        element = os.listdir(path)
    else:
        element = ['plot']
    if 'plot' not in element:
        os.system('mkdir plot')
    if measure == 'loss':
        fout = open('plot/' + file, 'w')
        fout.write('LOSS FUNCTION\n')
        for i in range(len(data)):
            fout.write(str(data[i]) + ' ')
        fout.write('\n')
        fout.close()
    elif measure == 'accuracy':
        fout = open('plot/' + file, 'a')
        if type == 'val':
            fout.write('ACCURACY VALIDATION\n')
        elif type == 'test':
            fout.write('ACCURACY TEST\n')
        elif type == 'heu_base_solver':
            fout.write('ACCURACY HEURISTIC vs SOLVER\n')
        elif type == 'heu_base_greedy':
            fout.write('ACCURACY HEURISTIC vs GREEDY\n')
        fout.write(str(data))
        fout.write('\n')
        fout.close()
    elif measure == 'error':
        fout = open('plot/' + file, 'a')
        if type == 'val':
            fout.write('ERROR VALIDATION\n')
        elif type == 'test':
            fout.write('ERROR TEST\n')
        elif type == 'heu_base_solver':
            fout.write('ERROR HEURISTIC BASE vs SOLVER\n')
        elif type == 'heu_base_greedy':
            fout.write('ERROR HEURISTIC BASE vs GREEDY\n')
        for j in range(len(data)):
            fout.write(str(data[j]))
            fout.write('\n')
        fout.close()



def read_all(path, file):
    lines = open(path + file, 'r').read().split('\n')
    accuracy = [-1.0, -1.0, -1.0]
    error = [[-10.0, -10.0, -10.0], [-10.0, -10.0, -10.0], [-10.0, -10.0, -10.0]]
    zopt = []
    loss = []
    for i in range(len(lines)):
        if lines[i] == 'LOSS FUNCTION':
            i = i + 1
            losses = lines[i].split(' ')
            for j in range(len(losses)-1):
                loss.append(float(losses[j]))
        elif lines[i] == 'ACCURACY VALIDATION':
            i = i + 1
            accuracy[0] = float(lines[i])
        elif lines[i] == 'ACCURACY TEST':
            i = i + 1
            accuracy[1] = float(lines[i])
        elif lines[i] == 'ACCURACY HEURISTIC':
            i = i + 1
            accuracy[2] = float(lines[i])
        elif lines[i] == 'ERROR VALIDATION':
            i = i + 1
            error[0][0] = float(lines[i])
            i = i + 1
            error[0][1] = float(lines[i])
            i = i + 1
            error[0][2] = float(lines[i])
        elif lines[i] == 'ERROR TEST':
            i = i + 1
            error[1][0] = float(lines[i])
            i = i + 1
            error[1][1] = float(lines[i])
            i = i + 1
            error[1][2] = float(lines[i])
        elif lines[i] == 'ERROR HEURISTIC':
            i = i + 1
            error[2][0] = float(lines[i])
            i = i + 1
            error[2][1] = float(lines[i])
            i = i + 1
            error[2][2] = float(lines[i])
        elif lines[i] == 'ZOPT':
            for j in range(i+1, len(lines)-1):
                line = lines[j].split(' ')
                sol = []
                for k in range(len(line)):
                    sol.append(int(line[k]))
                zopt.append(sol)
            break
    return loss, accuracy, error, zopt


def plot_loss(loss, show, path):
    import matplotlib as mpl
    mpl.use('tkagg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    it = []
    for i in range(len(loss)):
        it.append(i)
    ax.plot(it, loss)
    ax.set(xlabel='Iteration', ylabel='Loss function',
           title='Loss function during training network')
    ax.grid()
    fig.savefig(path)
    if show:
        plt.show()


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path_model, input_size, output_size, gpu):
    import Bnet as net
    model = net.Network(input_size, output_size, gpu)
    model.load_state_dict(torch.load(path_model))
    return model


'''
def read_params():
    path = './sorted/unfix/'
    params = sys.argv
    if len(params) > 1:
        for i in range(len(params)):
            if params[i] == "--path":
                if len(params) > i+1:
                    path = params[i+1]
                else:
                    usage(True)
                    sys.exit()
            elif params[i] == "--help" or params[i] == "-h":
                usage(False)
                sys.exit()
    return path
'''


def main(verbose, path):
    # read all arguments
    if verbose:
        print('###\tStart reading data')
    start = timer()
    train, validation, test = dt.read_data(path)
    end = timer()
    if verbose:
        print('###\tTime for reading data: %.2f sec' % (end - start))
    return train, validation, test, path
