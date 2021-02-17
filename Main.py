train_folder = 'images_background'
val_folder = 'images_evaluation'
from load_imgs_one_shot import *
from get_batch_one_shot import *
from Train import *
from Test import *
import pickle
import os

### creating pickle for train set ###
print('-'*10, 'training set','-'*10)
X,y,c=load_imgs(train_folder)
save_path = 'data'
with open(os.path.join(save_path,"train.pickle"), "wb") as f:
    pickle.dump((X,c),f)
    
lang_dict = c
print(lang_dict)
    
### creating pickle for test set ###
print('-'*10, 'test set','-'*10)
X,y,c=load_imgs(val_folder)
with open(os.path.join(save_path,"val.pickle"), "wb") as f:
    pickle.dump((X,c),f)

### plotting some= images ###
plot_images('images_background/Sanskrit/character03/')

### Setting up the model ###
model = get_model((105,105,1))
plot_model(model, show_shapes=True, expand_nested=True)
model.load_weights('model_weights.h5') ### Comment this when learning from scratch ###

### Uncomment below code when training from scratch ###
# N_ways=20
# epochs=20000
# optimizer=Adam(learning_rate=0.00006)
# trials = 250
# batch_size=32
# loss_every = 20
# evaluate_every = 10
# model = train(model,
#  N_ways,
#   epochs,
#    optimizer,
#     trials,
#      batch_size,
#       loss_every, evaluate_every)
### Uncomment above code when training from scratch ###



### Testing and plotting the model ###
ways = np.arange(1, 30, 2)
trials = 450
val_accs, train_accs, nn_accs = test(model, ways, False, trials)
fig,ax = plt.subplots(1)
ax.plot(ways, val_accs, "m", label="Siamese(val set)")
ax.plot(ways, train_accs, "y", label="Siamese(train set)")
plt.plot(ways, nn_accs, label="Nearest neighbour")

ax.plot(ways, 100.0/ways, "g", label="Random guessing")
plt.xlabel("Number of possible classes in one-shot tasks")
plt.ylabel("% Accuracy")
plt.title("Omiglot One-Shot Learning Performance of a Siamese Network")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
inputs,targets = make_one_shot_task(20, "val")
plt.show()