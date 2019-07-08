from timeit import default_timer as timer
import data as dt
import torch
import datetime
import copy
import random as rndm
import Bnet as net_opt
import Pnet as net_sol


def initialize(path, name):
    ##
    # Function that initialize output file
    ##
    f = open(path + name, 'w')
    f.write(str(datetime.datetime.now()) + '\n')
    f.close()


def usage(err):
    ##
    # Function that write parameters
    ##
    if err:
        print('Error in --path. Use:')
    print('<program.py> --help | --path path | --v')


def write_sol(data, tp, path, file, tp_heu):
    ##
    # Function that write greedy and heuristic solutions
    ##
    if path == '':
        print('Wrong path for write sol tp %d' % tp)
    else:
        f_out = open(path + file, 'a')
        # GREEDY
        if tp == 0:
            f_out.write('GREEDY SOLUTION\n')
            for j in range(len(data)):
                for i in range(len(data[j])):
                    if i != len(data[j]) - 1:
                        f_out.write(str(data[j][i]) + ' ')
                    else:
                        f_out.write(str(data[j][i]))
                f_out.write('\n')
        # HEURISTIC BASE
        elif tp == 1:
            if tp_heu == 'base':
                f_out.write('HEURISTIC BASE SOLUTION\n')
            elif tp_heu == 'incumbent':
                f_out.write('HEURISTIC INCUMBENT SOLUTION\n')
            elif tp_heu == 'best_incumbent':
                f_out.write('HEURISTIC BEST INCUMBENT SOLUTION\n')
            elif tp_heu == 'inv_greedy':
                f_out.write('INVERSE GREEDY SOLUTION\n')
            elif tp_heu == 'heu_rand':
                f_out.write('HEURISTIC BEST INCUMBENT SOLUTION --- RANDOM ITEMS ---\n')
            elif tp_heu == 'heu_prob':
                f_out.write('HEURISTIC BEST INCUMBENT SOLUTION --- PROBABILITY ITEMS ---\n')
            for j in range(len(data)):
                for i in range(len(data[j])):
                    if i != len(data[j]) - 1:
                        f_out.write(str(data[j][i]) + ' ')
                    else:
                        f_out.write(str(data[j][i]))
                f_out.write('\n')
        f_out.close()


def read_sol(path, file, n):
    ##
    # Function that read greedy and heuristic solutions
    ##
    gr = []
    heu_base = []
    heu_inc = []
    heu_b_inc = []
    heu_rand = []
    heu_prob = []
    inv_gr = []
    lines = open(path + file, 'r').read().split('\n')
    for i in range(len(lines)):
        if lines[i] == 'GREEDY SOLUTION':
            for j in range(i + 1, i + 1 + n):
                line = lines[j].split(' ')
                tmp = []
                for k in range(len(line)):
                    tmp.append((int(line[k])))
                gr.append(tmp)
            i += n
        elif lines[i] == 'HEURISTIC BASE SOLUTION':
            for j in range(i + 1, i + 1 + n):
                line = lines[j].split(' ')
                tmp = []
                for k in range(len(line)):
                    tmp.append((int(line[k])))
                heu_base.append(tmp)
            i += n
        elif lines[i] == 'HEURISTIC INCUMBENT SOLUTION':
            for j in range(i + 1, i + 1 + n):
                line = lines[j].split(' ')
                tmp = []
                for k in range(len(line)):
                    tmp.append((int(line[k])))
                heu_inc.append(tmp)
            i += n
        elif lines[i] == 'HEURISTIC BEST INCUMBENT SOLUTION':
            for j in range(i + 1, i + 1 + n):
                line = lines[j].split(' ')
                tmp = []
                for k in range(len(line)):
                    tmp.append((int(line[k])))
                heu_b_inc.append(tmp)
            i += n
        elif lines[i] == 'INVERSE GREEDY SOLUTION':
            for j in range(i + 1, i + 1 + n):
                line = lines[j].split(' ')
                tmp = []
                for k in range(len(line)):
                    tmp.append((int(line[k])))
                inv_gr.append(tmp)
            i += n
        elif lines[i] == 'HEURISTIC BEST INCUMBENT SOLUTION --- RANDOM ITEMS ---':
            for j in range(i + 1, i + 1 + n):
                line = lines[j].split(' ')
                tmp = []
                for k in range(len(line)):
                    tmp.append((int(line[k])))
                heu_rand.append(tmp)
            i += n
        elif lines[i] == 'HEURISTIC BEST INCUMBENT SOLUTION --- PROBABILITY ITEMS ---':
            for j in range(i + 1, i + 1 + n):
                line = lines[j].split(' ')
                tmp = []
                for k in range(len(line)):
                    tmp.append((int(line[k])))
                heu_prob.append(tmp)
            i += n
    return gr, heu_base, heu_inc, heu_b_inc, inv_gr, heu_rand, heu_prob


def write_all(data, path, file, measure, tp, flag, neg, pos):
    ##
    # Function that write error and accuracy
    ##
    f_out = open(path + file, 'a')
    if measure == 'loss':
        f_out.write('LOSS FUNCTION\n')
        for i in range(len(data)):
            f_out.write(str(data[i]) + ' ')
        f_out.write('\n')
        f_out.close()
    elif measure == 'accuracy':
        if tp == 'val':
            f_out.write('ACCURACY VALIDATION\n')
        elif tp == 'test':
            f_out.write('ACCURACY TEST\n')
        elif tp == 'heu_base_solver':
            f_out.write('ACCURACY HEURISTIC BASE vs SOLVER\n')
        elif tp == 'heu_base_greedy':
            f_out.write('ACCURACY HEURISTIC BASE vs GREEDY\n')
        elif tp == 'heu_inc_solver':
            f_out.write('ACCURACY HEURISTIC INCUMBENT vs SOLVER\n')
        elif tp == 'heu_inc_greedy':
            f_out.write('ACCURACY HEURISTIC INCUMBENT vs GREEDY\n')
        elif tp == 'heu_b_inc_solver':
            f_out.write('ACCURACY HEURISTIC BEST INCUMBENT vs SOLVER\n')
        elif tp == 'heu_b_inc_greedy':
            f_out.write('ACCURACY HEURISTIC BEST INCUMBENT vs GREEDY\n')
        elif tp == 'heu_rand_solver':
            f_out.write('ACCURACY HEURISTIC BEST INCUMBENT vs SOLVER --- RANDOM ITEMS ---\n')
        elif tp == 'heu_rand_greedy':
            f_out.write('ACCURACY HEURISTIC BEST INCUMBENT vs GREEDY --- RANDOM ITEMS ---\n')
        elif tp == 'heu_prob_solver':
            f_out.write('ACCURACY HEURISTIC BEST INCUMBENT vs SOLVER --- PROBABILITY ITEMS ---\n')
        elif tp == 'heu_prob_greedy':
            f_out.write('ACCURACY HEURISTIC BEST INCUMBENT vs GREEDY --- PROBABILITY ITEMS ---\n')
        elif tp == 'inv_solver':
            f_out.write('ACCURACY INVERSE GREEDY vs SOLVER\n')
        elif tp == 'inv_greedy':
            f_out.write('ACCURACY INVERSE vs GREEDY\n')
        f_out.write(str(data))
        f_out.write('\n')
        f_out.close()
    elif measure == 'error':
        if flag:
            if tp == 'heu_base_solver':
                f_out.write('ERROR HEURISTIC BASE vs SOLVER\n')
            elif tp == 'heu_base_greedy':
                f_out.write('ERROR HEURISTIC BASE vs GREEDY\n')
            elif tp == 'heu_inc_solver':
                f_out.write('ERROR HEURISTIC INCUMBENT vs SOLVER\n')
            elif tp == 'heu_inc_greedy':
                f_out.write('ERROR HEURISTIC INCUMBENT vs GREEDY\n')
            elif tp == 'heu_b_inc_solver':
                f_out.write('ERROR HEURISTIC BEST INCUMBENT vs SOLVER\n')
            elif tp == 'heu_b_inc_greedy':
                f_out.write('ERROR HEURISTIC BEST INCUMBENT vs GREEDY\n')
            elif tp == 'heu_rand_solver':
                f_out.write('ERROR HEURISTIC BEST INCUMBENT vs SOLVER --- RANDOM ITEMS ---\n')
            elif tp == 'heu_rand_greedy':
                f_out.write('ERROR HEURISTIC BEST INCUMBENT vs GREEDY --- RANDOM ITEMS ---\n')
            elif tp == 'heu_prob_solver':
                f_out.write('ERROR HEURISTIC BEST INCUMBENT vs SOLVER --- PROBABILITY ITEMS ---\n')
            elif tp == 'heu_prob_greedy':
                f_out.write('ERROR HEURISTIC BEST INCUMBENT vs GREEDY --- PROBABILITY ITEMS ---\n')
            elif tp == 'inv_solver':
                f_out.write('ERROR INVERSE vs SOLVER\n')
            elif tp == 'inv_greedy':
                f_out.write('ERROR INVERSE GREEDY vs GREEDY\n')
            f_out.write('MIN: %f\n' % data[0])
            f_out.write('MAX: %f\n' % data[1])
            f_out.write('AVG: %f\n' % data[2])
            f_out.write('AVG_NEG: %f %d\n' % (data[3], neg))
            f_out.write('AVG_POS: %f %d\n' % (data[4], pos))
        else:
            if tp == 'val':
                f_out.write('ERROR VALIDATION\n')
            elif tp == 'test':
                f_out.write('ERROR TEST\n')
            for j in range(len(data)):
                f_out.write(str(data[j]))
            f_out.write('\n')
    f_out.close()


def read_all(path, file):
    ##
    # Function that read error, accuracy
    ##
    lines = open(path + file, 'r').read().split('\n')
    accuracy = [-1.0, -1.0, -1.0]
    error = [[-10.0, -10.0, -10.0], [-10.0, -10.0, -10.0], [-10.0, -10.0, -10.0]]
    z_opt = []
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
        elif lines[i] == 'Z_OPT':
            for j in range(i+1, len(lines)-1):
                line = lines[j].split(' ')
                sol = []
                for k in range(len(line)):
                    sol.append(int(line[k]))
                z_opt.append(sol)
            break
    return loss, accuracy, error, z_opt


def plot_loss(loss, show, path):
    ##
    # Function that plot loss function
    ##
    import matplotlib as mpl
    mpl.use('tkagg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    it = []
    for i in range(len(loss)):
        it.append(i)
    ax.plot(it, loss)
    ax.set(xlabel='itation', ylabel='Loss function',
           title='Loss function during training network')
    ax.grid()
    fig.savefig(path)
    if show:
        plt.show()


def save_model(model, path):
    ##
    # Function that save the model
    ##
    torch.save(model.state_dict(), path)


def load_model(path_model, input_size, output_size, gpu, mod):
    ##
    # Function that load the model
    ##
    model = []
    if mod == 'opt':
        if gpu:
            model = net_opt.Network(input_size, output_size, gpu)
            model.load_state_dict(torch.load(path_model))
        else:
            model = net_opt.Network(input_size, output_size, gpu)
            model.load_state_dict(torch.load(path_model, map_location='cpu'))
    elif mod == 'solution':
        if gpu:
            model = net_sol.Network(input_size, output_size, gpu)
            model.load_state_dict(torch.load(path_model))
        else:
            model = net_sol.Network(input_size, output_size, gpu)
            model.load_state_dict(torch.load(path_model, map_location='cpu'))
    return model


'''
def read_params():
    ##
    # Function that read input parameters
    ##
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
    ##
    # Function that read all data
    ##
    # read all arguments
    if verbose:
        print('###\tStart reading data')
    start = timer()
    train, validation, test = dt.read_data(path)
    end = timer()
    if verbose:
        print('###\tTime for reading data: %.2f sec' % (end - start))
    return train, validation, test, path


def sorting_item(items, dec):
    ##
    # Function that sort items
    ##
    idx = []
    for i in range(len(items)):
        idx.append(i)

    for k in range(len(items)):
        # sorting profit / weight ratio

        for j in range(k + 1, len(items)):
            if dec:
                if items[k][0] < items[j][0]:
                    idx[j] = k
                    idx[k] = j
                    tmp_w = items[k][0]
                    tmp_p = items[k][1]
                    items[k][0] = items[j][0]
                    items[k][1] = items[j][1]
                    items[j][0] = tmp_w
                    items[j][1] = tmp_p
            else:
                if items[k][0] > items[j][0]:
                    tmp = idx[k]
                    idx[k] = idx[j]
                    idx[j] = tmp
                    tmp_w = items[k][0]
                    tmp_p = items[k][1]
                    items[k][0] = items[j][0]
                    items[k][1] = items[j][1]
                    items[j][0] = tmp_w
                    items[j][1] = tmp_p
    return items, idx


def sort_weight(test, it, dec):
    ##
    # Function that sort items based on profit / weight ratio
    ##

    # BRUTE FORCE SORTING -> OPTIMIZE!
    idx = []
    for i in range(test[it].n_item):
        idx.append(i)
    items = copy.deepcopy(test[it].items)

    for k in range(len(items)):
        # sorting profit / weight ratio
        for j in range(k + 1, len(items)):
            if dec:
                if items[k][0] < items[j][0]:
                    idx[j] = k
                    idx[k] = j
                    tmp_w = items[k][0]
                    tmp_p = items[k][1]
                    items[k][0] = items[j][0]
                    items[k][1] = items[j][1]
                    items[j][0] = tmp_w
                    items[j][1] = tmp_p
            else:
                if items[k][0] > items[j][0]:

                    tmp = idx[k]
                    idx[k] = idx[j]
                    idx[j] = tmp
                    tmp_w = items[k][0]
                    tmp_p = items[k][1]
                    items[k][0] = items[j][0]
                    items[k][1] = items[j][1]
                    items[j][0] = tmp_w
                    items[j][1] = tmp_p
    return items, idx


def sort(test, it, dec):
    ##
    # Function that sort items based on profit / weight ratio
    ##

    # BRUTE FORCE SORTING -> OPTIMIZE!
    items = []
    for k in range(test[it].n_item):
        # sorting profit / weight ratio
        items = test[it].items
        for j in range(k + 1, test[it].n_item):
            if dec:
                if items[k][1] / items[k][0] < items[j][1] / items[j][0]:
                    tmp_w = items[k][0]
                    tmp_p = items[k][1]
                    items[k][0] = items[j][0]
                    items[k][1] = items[j][1]
                    items[j][0] = tmp_w
                    items[j][1] = tmp_p
            else:
                if items[k][1] / items[k][0] > items[j][1] / items[j][0]:
                    tmp_w = items[k][0]
                    tmp_p = items[k][1]
                    items[k][0] = items[j][0]
                    items[k][1] = items[j][1]
                    items[j][0] = tmp_w
                    items[j][1] = tmp_p
    return items


def partial_greedy(capacity, items):
    ##
    # Function that compute partial greedy solution
    ##
    opt_sol = []
    total_w = 0
    for j in range(len(items)):
        if total_w + items[j][0] <= capacity:
            opt_sol.append(j)
            total_w += items[j][0]
    return opt_sol


def greedy(data, it):
    ##
    # Function that compute greedy solutions
    ##
    items = sort(data, it, True)
    opt_sol = []
    total_w = 0
    for j in range(data[it].n_item):
        if total_w + items[j][0] <= data[it].capacity:
            opt_sol.append(j)
            total_w += items[j][0]
    return opt_sol


def inv_greedy(data, it):
    ##
    # Function that compute inverse greedy solutions
    ##
    items, idx = sort_weight(data, it, False)
    opt_sol = []
    total_w = 0
    for j in range(data[it].n_item):
        if total_w + items[j][0] <= data[it].capacity:
            opt_sol.append(j)
            total_w += items[j][0]
    for k in range(len(opt_sol)):
        opt_sol[k] = idx[opt_sol[k]]
    return opt_sol


def compute_z(test, sol, it):
    ##
    # Function that compute value of z
    ##
    z = 0
    for i in range(len(sol)):
        z += test[it].items[sol[i]][1]
    return z


def create_items(data, it, maybe, start, inv):
    ##
    # Function that create separate items for computing solutions
    ##
    items = []
    idx = []
    for i in range(start, len(maybe)):
        items.append(data[it].items[maybe[i]].copy())
    if inv:
        items, idx = sorting_item(items, False)
    return items, idx


def heuristic_base(data, epoch, input_size, network):
    ##
    # Function that compute heuristic solutions
    ##
    maybe = []
    # compute total of weights
    total_w = 0
    for i in range(data[epoch].n_item):
        total_w = total_w + data[epoch].items[i][0]

    # compute partial weights
    partial_w = 0
    opt_sol = []

    for j in range(data[epoch].n_item):
        partial_w = partial_w + data[epoch].items[j][0]

        # input_in & input_out
        input_in = [0] * input_size
        input_out = [0] * input_size
        # prepare input for NN
        for i in range(len(opt_sol)):
            item_w = data[epoch].items[opt_sol[i]][0] * (partial_w/total_w) / data[epoch].capacity
            input_in[opt_sol[i] * 2 + 1] = data[epoch].items[opt_sol[i]][1] / data[epoch].ub_dan
            input_in[opt_sol[i] * 2] = item_w
            input_out[opt_sol[i] * 2 + 1] = data[epoch].items[opt_sol[i]][1] / data[epoch].ub_dan
            input_out[opt_sol[i] * 2] = item_w
        item_w = data[epoch].items[j][0] * (partial_w / total_w)
        input_in[j * 2 + 1] = data[epoch].items[j][1] / data[epoch].ub_dan
        input_in[j * 2] = item_w

        if network.gpu:
            inputs_in = torch.cuda.FloatTensor([input_in])
            inputs_out = torch.cuda.FloatTensor([input_out])
        else:
            inputs_in = torch.Tensor([input_in])
            inputs_out = torch.Tensor([input_out])
        # output of NN
        output_in = network(inputs_in)
        output_out = network(inputs_out)
        # chose item
        if float(output_in) > float(output_out):
            maybe.append(j)
    # compute greedy solution
    items, _ = create_items(data, epoch, maybe, 0, False)
    opt_sol = partial_greedy(data[epoch].capacity, items)
    for i in range(len(opt_sol)):
        opt_sol[i] = maybe[opt_sol[i]]

    return opt_sol, maybe


def heuristic_inc(data, epoch, input_size, network, maybe):
    ##
    # Function that compute heuristic solutions with greedy incumbent
    ##

    # compute greedy solution and NN items
    if len(maybe) == 0:
        _, maybe = heuristic_base(data, epoch, input_size, network)

    # initialize incumbent solution
    items, _ = create_items(data, epoch, maybe, 0, False)
    incumbent_sol = partial_greedy(data[epoch].capacity, items)
    for i in range(len(incumbent_sol)):
        incumbent_sol[i] = maybe[incumbent_sol[i]]
    incumbent = compute_z(data, incumbent_sol, epoch)

    for i in range(1, len(maybe)):
        # compute actual solution for NN items
        items, _ = create_items(data, epoch, maybe, i, False)
        tmp_sol = copy.deepcopy(partial_greedy(data[epoch].capacity, items))
        for j in range(len(tmp_sol)):
            tmp_sol[j] += i
            tmp_sol[j] = maybe[tmp_sol[j]]
        z_tmp = compute_z(data, tmp_sol, epoch)
        # update incumbent solution
        if z_tmp > incumbent:
            incumbent = copy.deepcopy(z_tmp)
            incumbent_sol = tmp_sol

    return incumbent, incumbent_sol


def heuristic_b_inc(data, epoch, input_size, network):
    ##
    # Function that compute heuristic solutions with greedy incumbent
    ##

    # compute greedy solution and NN items
    _, maybe = heuristic_base(data, epoch, input_size, network)
    incumbent, incumbent_sol = heuristic_inc(data, epoch, input_size, network, maybe)
    # initialize incumbent solution
    inv_gr_sol = inv_greedy(data, epoch)
    z_inv_gr = compute_z(data, inv_gr_sol, epoch)

    if z_inv_gr > incumbent:
        incumbent = z_inv_gr
        incumbent_sol = inv_gr_sol

    for i in range(len(maybe)):
        # compute actual solution for NN items
        inv_items, idx = create_items(data, epoch, maybe, i, True)
        tmp_inv_sol = partial_greedy(data[epoch].capacity, inv_items)
        for j in range(len(tmp_inv_sol)):
            tmp_inv_sol[j] = maybe[idx[tmp_inv_sol[j]] + i]

        z_inv_tmp = compute_z(data, tmp_inv_sol, epoch)

        if z_inv_tmp > incumbent:
            incumbent = z_inv_tmp
            incumbent_sol = tmp_inv_sol
    return incumbent, incumbent_sol


def heuristic_rand(data, epoch, input_size, network):
    ##
    # Function that compute heuristic solutions with greedy incumbent
    ##

    # compute greedy solution for random items
    maybe = []
    for i in range(data[epoch].n_item):
        if rndm.random() > 0.5:
            maybe.append(i)

    incumbent, incumbent_sol = heuristic_inc(data, epoch, input_size, network, maybe)
    # initialize incumbent solution
    inv_gr_sol = inv_greedy(data, epoch)
    z_inv_gr = compute_z(data, inv_gr_sol, epoch)

    if z_inv_gr > incumbent:
        incumbent = z_inv_gr
        incumbent_sol = inv_gr_sol

    for i in range(len(maybe)):
        # compute actual solution for NN items
        inv_items, idx = create_items(data, epoch, maybe, i, True)
        tmp_inv_sol = partial_greedy(data[epoch].capacity, inv_items)
        for j in range(len(tmp_inv_sol)):
            tmp_inv_sol[j] = maybe[idx[tmp_inv_sol[j]] + i]

        z_inv_tmp = compute_z(data, tmp_inv_sol, epoch)

        if z_inv_tmp > incumbent:
            incumbent = z_inv_tmp
            incumbent_sol = tmp_inv_sol
    return incumbent, incumbent_sol


def heuristic_b_inc_pr(data, epoch, input_size, network, maybe):
    ##
    # Function that compute heuristic solutions with greedy incumbent
    ##

    # compute greedy solution and NN items
    incumbent, incumbent_sol = heuristic_inc(data, epoch, input_size, network, maybe)
    # initialize incumbent solution
    inv_gr_sol = inv_greedy(data, epoch)
    z_inv_gr = compute_z(data, inv_gr_sol, epoch)

    if z_inv_gr > incumbent:
        incumbent = z_inv_gr
        incumbent_sol = inv_gr_sol

    for i in range(len(maybe)):
        # compute actual solution for NN items
        inv_items, idx = create_items(data, epoch, maybe, i, True)
        tmp_inv_sol = partial_greedy(data[epoch].capacity, inv_items)
        for j in range(len(tmp_inv_sol)):
            tmp_inv_sol[j] = maybe[idx[tmp_inv_sol[j]] + i]

        z_inv_tmp = compute_z(data, tmp_inv_sol, epoch)

        if z_inv_tmp > incumbent:
            incumbent = z_inv_tmp
            incumbent_sol = tmp_inv_sol
    return incumbent, incumbent_sol


def check_heu_sol(heuristic, data):
    ##
    # Function that check feasibility of heuristic solutions
    ##
    if len(heuristic) > 0:
        for i in range(len(heuristic)):
            weight = 0
            for j in range(len(heuristic[i])):
                weight += data[i].items[heuristic[i][j]][0]

            if weight > data[i].capacity:
                return -1
    return 0


def error_accuracy(targets, error, accuracy, z_heu, better, it):
    ##
    # Function that compute error and accuracy for heuristic
    ##
    if (z_heu[it] - targets) / targets < error[0]:
        better[0] += 1
        error[0] = (z_heu[it] - targets) / targets
    elif (z_heu[it] - targets) / targets > error[1]:
        better[1] += 1
        error[1] = (z_heu[it] - targets) / targets

    error[2] = error[2] + abs((z_heu[it] - targets) / targets)

    if z_heu[it] == targets:
        accuracy = accuracy + 1

    if z_heu[it] - targets < 0:
        better[2] += 1
        error[3] += (z_heu[it] - targets) / targets
    elif z_heu[it] - targets > 0:
        better[3] += 1
        error[4] += (z_heu[it] - targets) / targets

    return accuracy, error, better


def heuristic_performance(data, compare, heuristic, divide, tp):
    ##
    # Function that compare error and accuracy between heuristic / solver / greedy solutions
    ##
    z_heu = []
    error = [0, 0, 0, 0, 0]
    accuracy = 0
    better = [0, 0, 0, 0]

    for i in range(len(heuristic)):
        if tp == 'solver':
            targets = data[i].z_opt
        else:
            targets = compute_z(data, compare[i], i)

        z_heu.append(compute_z(data, heuristic[i], i))
        accuracy, error, better = error_accuracy(targets, error, accuracy, z_heu, better, i)

    accuracy = accuracy / divide
    for i in range(len(better)):
        if better[i] != 0:
            error[i + 1] = error[i + 1] / better[i]
    error[2] = error[2] / divide

    return accuracy, error, better


def compute_feasibility(data, epoch, solution):
    ##
    # Function that compute solution feasible
    ##
    weight = 0
    for i in range(len(solution)):
        weight += data[epoch].items[solution[i]][0]
    if weight > data[epoch].capacity:
        return -1
    else:
        return 0
