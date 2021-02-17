import numpy as np
from sklearn.utils import shuffle
from load_tensors import *
def make_one_shot_task(N, s='val', language=None):
    Xtrain, train_classes, Xtest, val_classes = load_tensor_data()
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xtest
        categories = val_classes

    n_classes, n_examples, w, h = X.shape
    
    rng = np.random.default_rng()
    if language is not None:
        high, low = categories[language]
        if N > high-low:
            raise ValueError(language, 'has less than {} letters'.format(N))
        categories = rng.choice(range(low, high), (N,), replace=False) ### Select N chars ###
    else:
        categories = rng.choice(range(n_classes), (N,), replace=False)
    
    true_category = categories[0]
    ex1, ex2 = rng.choice(n_examples, (2, ), replace=False) 
    indices = np.random.randint(0, n_examples, size=(N,))    ### Select N images ###
    test_image_set = np.array([X[true_category,ex1,:,:]]*N).reshape(N,w,h,1)
    support_set = X[categories,indices,:,:] ### Select N images one from each previously chosen N char ###
    support_set[0,:,:] = X[true_category, ex2]
    support_set.reshape(N,w,h,1)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image_set, support_set = shuffle(targets, test_image_set, support_set)
    pairs = [test_image_set, support_set]
    return pairs, targets


def nearest_neighbour_correct(pairs, targets):
    L2distance = np.zeros_like(targets)
    for i in range(len(targets)):
        L2distance[i] = np.sum(np.sqrt(pairs[0][i]**2 - pairs[1][i]**2))  ### this can be changed ###
    if np.argmin(L2distance) == np.argmax(targets):
        return 1
    else:
        return 0


def test_oneshot(model, N, k, s='val', verbose=0, model_type='siamese'):
    n_correct = 0
    if verbose:
        print('{} random {} way one-shot learning'.format(k,N))
    for i in range(k):
        inputs, target = make_one_shot_task(N, s)
        if model_type == 'siamese':
            prob = model.predict(inputs)
            if np.argmax(prob) == np.argmax(target):
                n_correct += 1
        else:
            n_correct += nearest_neighbour_correct(inputs, target)
    percent_correct = (n_correct * 100) / k
    if verbose:
        if model_type=='siamese':
            print('evaluation for siamese')
        else:
            print('evaluation for nearest neighbour')
        print('{}% correct for {}-way shot learning'.format(percent_correct,N))

    return percent_correct    