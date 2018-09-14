'''Data generators for various tasks to test the relational rnn model.
'''
import numpy as np
import keras
from scipy.spatial.distance import pdist, squareform

def generate_nthfarthest_data(n_samples, n_items, n_dims):
    '''What is the n-th farthest vector (in Euclidean distance) from vector m'''
    x = 2. * np.random.rand(n_samples, n_items, n_dims) - 1.
    l = np.argsort(np.random.randn(n_samples, n_items), axis=-1)
    l_onehot = np.reshape(keras.utils.to_categorical(l.flatten(), num_classes=n_items), (n_samples, n_items, n_items))

    n = np.random.randint(low=0, high=n_items, size=(n_samples,)) # n-th farthest
    m = np.random.randint(low=0, high=n_items, size=(n_samples,)) # target index, m

    n_onehot = keras.utils.to_categorical(n, num_classes=n_items)
    m_onehot = keras.utils.to_categorical(m, num_classes=n_items)

    n_onehot = np.repeat(np.reshape(n_onehot, (n_samples, 1, n_items)), n_items, axis=1)
    m_onehot = np.repeat(np.reshape(m_onehot, (n_samples, 1, n_items)), n_items, axis=1)

    X = np.concatenate((x,l_onehot,n_onehot,m_onehot),axis=-1)
    y = np.zeros(n_samples)

    for i in range(n_samples):
        p = np.argsort(squareform(pdist(x[i,:,:], 'euclidean')))
        y[i] = l[i, p[m[i]==l[i], -n[i]][0]]  # target vector label

    Y = keras.utils.to_categorical(y, num_classes=n_items)

    return X, Y