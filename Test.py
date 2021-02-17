import numpy as np
import matplotlib.pyplot as plt
from make_and_test_one_shot import *
from siamese_model_one_shot import *

def test(model, ways=np.arange(1, 30, 2), resume=False, trials=450):
    val_accs, train_accs,nn_accs = [], [], []
    for N in ways:
        val_accs.append(test_oneshot(model, N,trials, "val", verbose=True))
        train_accs.append(test_oneshot(model, N,trials, "train", verbose=True))
        nn_accs.append(test_oneshot(model, N,trials, model_type='nn', verbose=True))
        
    return val_accs, train_accs, nn_accs