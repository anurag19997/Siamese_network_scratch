import numpy as np
from load_tensors import *
def get_batch(batch_size, s='train'):
    Xtrain, train_classes, Xtest, val_classes = load_tensor_data()
    if s == 'train':
        X = Xtrain
    else:
        X = Xtest
    n_classes, n_examples, h, w = X.shape
    rng = np.random.default_rng()
    categories = rng.choice(n_classes, batch_size, replace=False) ### selecting from output classes ###

    pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]

    targets = np.zeros((batch_size,))
    targets[batch_size//2:] = 1 
    
    for i in range(batch_size):
        category = categories[i]
        idx1 = rng.integers(0, n_examples)
        pairs[0][i,:,:,:] = X[category, idx1].reshape(w,h,1)
        idx2 = rng.integers(0, n_examples)
        
        if i > batch_size//2:
            category2 = category
        else:
            category2 = (category + rng.integers(1, n_classes)) % n_classes
        
        pairs[1][i,:,:,:] = X[category2, idx2].reshape(w,h,1)
        
    return pairs, targets

def generator(batch_size, s='train'):
    while True:
        pairs, targets = get_batch(batch_size, s)
        yield(pairs, targets)
        