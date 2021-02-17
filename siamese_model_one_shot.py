from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Lambda
from keras import Input, Model
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model
from keras.optimizers import Adam
from model_initializer import *
def get_model(input_shape):
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    model = Sequential()
    
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape = input_shape,  ### First Layer ###
                     kernel_initializer = initialize_weights,
                     kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(128, (7,7), activation='relu',                              ### Second Layer ###
                    kernel_initializer = initialize_weights,
                    bias_initializer = initialize_bias, 
                    kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(128, (4,4), activation='relu',                             ### third Layer ###
                    kernel_initializer = initialize_weights,
                    bias_initializer = initialize_bias, 
                    kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(256, (4,4), activation='relu',                            ### fourth Layer ###
                    kernel_initializer = initialize_weights,
                    bias_initializer = initialize_bias, 
                    kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096,
              activation='sigmoid',
              kernel_initializer = initialize_weights,
              bias_initializer = initialize_bias, 
              kernel_regularizer=l2(1e-3)))
    
    encodedl = model(left_input)
    encodedr = model(right_input)
    
    ### Customized layer for getting the difference between two encodings ###
    L1Layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1Distance = L1Layer([encodedl, encodedr])
    
    ### Final Layer ###
    prediction = Dense(1, activation='sigmoid', bias_initializer = initialize_bias)(L1Distance)
    
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    print(siamese_net.summary())
    
    return siamese_net