from get_batch_one_shot import *
import time
from make_and_test_one_shot import *
from siamese_model_one_shot import *


### Training code ###
def train(model, N_ways=20, epochs=20000, optimizer=Adam(learning_rate=0.00006), trials = 250, batch_size=32, loss_every = 20, evaluate_every = 10):
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    start_time = time.time()
    best = -1
    for i in range(1, epochs):
        inputs, targets = get_batch(batch_size)
        loss = model.train_on_batch(inputs, targets)
        print('\n','-'*20)
        print('Loss {}'.format(loss))
        if i % evaluate_every == 0:
            t = time.time() - start_time
            print('Time for {} iterations: {:.2f} seconds'.format(i, t))
            val_acc = test_oneshot(model, N_ways, trials, s='train', verbose=True)
            if val_acc >= best:
                print('Current best:{} and previous best:{}'.format(val_acc, best))
                best = val_acc
        if i % loss_every == 0:
            print('Loss for {} iteration is {:.2f}'.format(i, loss))

    model.save_weights('data/model_weights_updates.h5')
    return model