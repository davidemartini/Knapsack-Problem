import torch
import torch.nn as nn
from torch.autograd import Variable
import string
import time
import math
import os
import random
import rnn as model


class Shakespeare:
    # Initialization
    def __init__(self, filename, chunk_len, batch_size, gpu, model_saved, hidden_size, n_layers, learning_rate, print_every, n_epochs, prime_str, predict_len, temperature):
        super(Shakespeare, self).__init__()
        self.all_characters = string.printable
        self.n_characters = len(self.all_characters)
        self.filename = filename
        self.file, self.file_len = self.read_file()
        self.start = time.time()
        self.chunk_len = chunk_len
        self.batch_size = batch_size
        self.gpu = gpu
        self.model_saved = model_saved

        # Create the RNN
        self.decoder = model.CharRNN1(self.n_characters, hidden_size, self.n_characters, n_layers)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        if self.gpu:
            self.decoder.cuda()

        # Create input and target
        self.inp = torch.LongTensor(self.batch_size, self.chunk_len)
        self.target = torch.LongTensor(self.batch_size, self.chunk_len)
        self.print_every = print_every
        self.n_epochs = n_epochs
        self.predicted = ""
        self.prime_str = prime_str
        self.predict_len = predict_len
        self.temperature = temperature

    def read_file(self):
        # Read the file
        file = open(self.filename).read()
        file_len = len(file)
        return file, file_len

    def char_tensor(self, str):
        # Turning a string into a tensor
        tensor = torch.zeros(len(str)).long()
        for c in range(len(str)):
            tensor[c] = self.all_characters.index(str[c])
        return tensor

    def time_since(self):
        # Readable time elapsed
        s = time.time() - self.start

        # math.floor = cast to int
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def random_training_set(self):
        # Create the training set
        # Create group of data of batch_size
        for bi in range(self.batch_size):
            start_index = random.randint(0, self.file_len - self.chunk_len)
            end_index = start_index + self.chunk_len + 1
            chunk = self.file[start_index:end_index]

            # Create tensor for input and target
            self.inp[bi] = self.char_tensor(chunk[:-1])
            self.target[bi] = self.char_tensor(chunk[1:])
        '''
        Variable is a thin wrapper around a Tensor object, that also holds
        the gradient w.r.t. to it, and a reference to a function that created it.
        This reference allows retracing the whole chain of operations that
        created the data. If the Variable has been created by the user, its grad_fn
        will be  ``None`` and we call such objects *leaf* Variables.
        '''
        # Need for use gradient
        self.inp = Variable(self.inp)
        self.target = Variable(self.target)
        if self.gpu:
            self.inp = self.inp.cuda()
            self.target = self.target.cuda()

    def train(self):
        # Initialization
        hidden = self.decoder.init_hidden(self.batch_size)
        if self.gpu:
            hidden = hidden.cuda()
        self.decoder.zero_grad()
        loss = 0

        # Training
        for c in range(self.chunk_len):
            output, hidden = self.decoder(self.inp[:, c], hidden)
            loss += self.criterion(output.view(self.batch_size, -1), self.target[:, c])
        loss.backward()
        self.decoder_optimizer.step()
        return loss.data.item() / self.chunk_len

    def save(self):
        # Save the trained model in self.model_saved file
        save_filename = os.path.splitext(os.path.basename(self.model_saved))[0] + '.pt'
        torch.save(self.decoder, save_filename)
        print('Saved as %s' % save_filename)

    def main(self):
        # Start the train for n_epochs
        self.start = time.time()
        loss_avg = 0
        print("Training for %d epochs..." % self.n_epochs)
        for epoch in range(1, self.n_epochs + 1): # tqdm(range(1, n_epochs + 1)): tqdm show progress bar
            loss = self.train()
            loss_avg += loss
            if epoch % self.print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (self.time_since(), epoch, epoch / self.n_epochs * 100, loss))

    def predict(self):
        # Load the trained model, and delete it, and prepare the first input
        self.decoder = torch.load(self.model_saved)
        del self.model_saved
        hidden = self.decoder.init_hidden(1)
        prime_input = Variable(self.char_tensor(self.prime_str).unsqueeze(0))
        if self.gpu:
            hidden = hidden.cuda()
            prime_input = prime_input.cuda()
        self.predicted = self.prime_str

        # Use priming string to "build up" hidden state
        for p in range(len(self.prime_str) - 1):
            '''
            The underscore is also used for ignoring the specific values. 
            If you don’t need the specific values or the values are not used, just assign the values to underscore.
            '''
            _, hidden = self.decoder(prime_input[:, p], hidden)
        inp = prime_input[:, -1]

        # Strat prediction for predict_len chars
        for p in range(self.predict_len):
            output, hidden = self.decoder(inp, hidden)

            # Sample from the network as a multinomial distribution
            '''
            Returns a tensor where each row contains 1 indices 
            sampled from the multinomial probability distribution located in the corresponding row of tensor output_dist
            '''
            output_dist = output.data.view(-1).div(self.temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            # Add predicted character to string and use as next input
            predicted_char = self.all_characters[top_i]
            self.predicted += predicted_char
            inp = Variable(self.char_tensor(predicted_char).unsqueeze(0))
            if self.gpu:
                inp = inp.cuda()
