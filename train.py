'''Trains an LSTM or relational LSTM model on relational recurrent tasks.
'''
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from generate_data import generate_nthfarthest_data
from relational_recurrent import RLSTM

batch_size = 3200
n_items = 8
n_dims = 16
n_inp = n_dims + 3*n_items
memdim = n_inp  # hmmm ... input dimension must be the same as memory dimension
memsize = 16
n_hid = memdim * memsize  # total number of hidden units
n_epochs = 10000

print('Building the model...')
model = Sequential()
# model.add(LSTM(n_hid, input_shape=(n_items, n_inp)))
model.add(RLSTM(memdim, memsize, input_shape=(n_items, n_inp)))
model.add(Dense(n_items, activation='softmax'))

# set up the optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Training...')
for i in range(n_epochs):
    X, Y = generate_nthfarthest_data(batch_size, n_items, n_dims)
    score, acc = model.train_on_batch(X, Y)  # No need to evaluate separately because the data are generated on the fly
    print('Epoch: %i, Loss: %.4f, Test accuracy: %.4f'%(i,score,acc))