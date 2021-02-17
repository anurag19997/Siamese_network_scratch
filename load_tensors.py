import pickle
def load_tensor_data():
    data_path = 'data'
    with open("data/train.pickle", "rb") as f:
        (Xtrain, train_classes) = pickle.load(f)

    with open("data/val.pickle", "rb") as f:
        (Xtest, val_classes) = pickle.load(f)
        
    print("Training alphabets: \n")
    print(list(train_classes.keys()))
    print("\n\nValidation alphabets:", end="\n\n")
    print(list(val_classes.keys()))
    return Xtrain, train_classes, Xtest, val_classes