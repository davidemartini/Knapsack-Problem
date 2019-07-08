import os


# n  capacity_KP w[1] … w[n] p[1] … p[2] card sol[1] .. sol[card]  zheu  zopt uppper_bound_Dantzig_arrotondato_in_basso
class Data:
    # Initialization
    def __init__(self, capacity, n_item, item, card, solution, z_heu, z_opt, ub_dan):
        self.capacity = capacity
        self.card = card
        self.n_item = n_item
        self.z_heu = z_heu
        self.z_opt = z_opt
        self.ub_dan = ub_dan
        self.items = item
        self.solutions = solution


def make_data(lines):
    vett = []
    for j in range(len(lines) - 1):
        data = lines[j].split(' ')
        n_item = int(data[0])
        capacity = int(data[1])
        card = int(data[n_item * 2 + 2])
        index = n_item * 2 + 3
        solution = []
        for k in range(card):
            solution.append(int(data[index]) - 1)
            index = index + 1
        z_heu = int(data[index])
        index = index + 1
        z_opt = int(data[index])
        index = index + 1
        ub_dan = int(data[index])
        item = []
        w = 2
        p = w + n_item
        for k in range(n_item):
            item.append([int(data[w]), int(data[p])])
            p = p + 1
            w = w + 1
        vett.append(Data(capacity, n_item, item, card, solution, z_heu, z_opt, ub_dan))
    return vett


def read_data(path):
    files = os.listdir(path)
    train = []
    validation = []
    test = []

    for i in range(len(files)):
        file = open(path + files[i], 'r')
        lines = file.read().split('\n')
        file.close()
        tp = files[i].split('_')
        if tp[0] == 'train':
            # train set
            train = make_data(lines)
        elif tp[0] == 'validation':
            # validation set
            validation = make_data(lines)
        elif tp[0] == 'test':
            # test set
            test = make_data(lines)

    return train, validation, test
