import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
def load_imgs(path, n=0):
    '''
    path ==> path to images directory
    
    '''
    char_dict = {}
    lang_dict = {}
    X = []
    y = []
    curr_y = n
    
    for alpha in os.listdir(path):
        print('loading alphabet', alpha)
        lang_dict[alpha] = [curr_y, None]   ### getting first and last index of a alpha ###
        alpha_path = os.path.join(path, alpha)
        
        for char in os.listdir(alpha_path):
            char_dict[curr_y] = (alpha, char) ### which index belongs to which char and alpha ###
            char_path = os.path.join(alpha_path, char)
            category_imgs = []
            
            for img in os.listdir(char_path):
                category_imgs.append(cv2.imread(os.path.join(char_path,img))) ### 1. getting path, 2. reading img, 3. appending img to list ###
                y.append(curr_y) ### output labels are indices of char ###
            
            lang_dict[alpha][1] = curr_y ### updating last index of alpha ###
            X.append(np.stack(category_imgs)) ### stacking all the 20 imgs of a char ###
            curr_y += 1 ### index is updated for each char ###
        
    try:
        X = np.stack(X) ### stacking all the chars together ###
    except Exception as e:
        print(e)
        print("error - category_images:", len(category_imgs))
        
    y = np.vstack(y) ### stacking all the indices of chars together ###
    
    return X, y, lang_dict

def plot_images(path):
    fig = plt.figure(figsize=(10,10))
    row,col = 5,4
    i=1
    for img in os.listdir(path):
        img = cv2.imread(os.path.join(path,img))
        fig.add_subplot(row, col, i)
        i += 1
        plt.imshow(img)
    plt.show()