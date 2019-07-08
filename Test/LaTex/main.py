import latex as ltx

# Initialization of parameters
print_every = 100
filename = './def.tex'
hidden_size = 512
n_layers = 3
gpu = False
learning_rate = 0.01
n_epochs = 10000
batch_size = 50
chunk_len = 200
temperature = 0.8
predict_len = 500
prime_str = 'A'
model_saved = 'RNNlatex.pt'
net = 'RNN'
f = open("RNNresult.txt", "w")
# Initialize the RNN
sh = ltx.Latex(
    filename,
    chunk_len,
    batch_size,
    gpu,
    model_saved,
    hidden_size,
    n_layers,
    learning_rate,
    print_every,
    n_epochs,
    prime_str,
    predict_len,
    temperature,
    net)
# Create training set
sh.random_training_set()
# Train the net
sh.main()
# Save the trained model
sh.save()
# Try to predict some text
sh.predict()
#print(sh.predicted)
f.write("hidden size: %d\nbatch size: %d\nepochs %d\n" % (hidden_size, batch_size, n_epochs))
f.write(sh.predicted)
f.write("\n\n")
f.close()
