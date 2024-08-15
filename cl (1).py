import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
#%matplotlib inline
np.random.seed(0) # so our results will be copyable

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score, classification_report
from sklearn.metrics import classification_report

import seaborn as sns
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Conv1D, MaxPool1D, Input
from keras.optimizers import Adam, RMSprop
from keras.losses import SparseCategoricalCrossentropy

import mlxtend 
import hashlib
import zlib

import sys
import socket
import pickle

random_seed = 0

def load_data():
    unhealthy = pd.read_csv('C:/Users/vedav/OneDrive/Desktop/Final_Project/ptbdb_abnormal.csv', header=None)
    healthy = pd.read_csv('C:/Users/vedav/OneDrive/Desktop/Final_Project/ptbdb_normal.csv', header=None)
    healthy = healthy.rename({187: 'Label'}, axis=1)
    unhealthy = unhealthy.rename({187: 'Label'}, axis=1)
    healthy_x_train, healthy_x_test, healthy_y_train, healthy_y_test= train_test_split(healthy.iloc[:, :187], healthy['Label'], test_size=0.2, shuffle=True)
    unhealthy_x_train, unhealthy_x_test, unhealthy_y_train, unhealthy_y_test= train_test_split(unhealthy.iloc[:, :187], unhealthy['Label'], test_size=0.2, shuffle=True)
    
    frames=[healthy_x_train, unhealthy_x_train]
    train_x= pd.concat(frames)

    frames=[healthy_y_train, unhealthy_y_train]
    train_y= pd.concat(frames)

    frames=[healthy_x_test, unhealthy_x_test]
    test_x= pd.concat(frames)

    frames=[healthy_y_test, unhealthy_y_test]
    test_y= pd.concat(frames)
    
    healthy_x_train['Label'] = healthy_y_train
    unhealthy_x_train['Label'] = unhealthy_y_train
    print("Shape of Unhealthy_train: ",unhealthy_x_train.shape)
    print("Shape of Healthy_train: ",healthy_x_train.shape)
    
    healthy_upsample = resample(healthy_x_train, replace=True, n_samples=6000, random_state=random_seed+1)
    unhealthy_downsample = resample(unhealthy_x_train, replace=True, n_samples=6000, random_state=random_seed+1)
    
    frames=[healthy_upsample , unhealthy_downsample]
    ptb_dfs = pd.concat(frames)
    
    
    print(ptb_dfs['Label'].value_counts())
    
    
    x_train, y_train = ptb_dfs.iloc[:, :187], ptb_dfs['Label']
    healthy_x_test['Label']=healthy_y_test
    unhealthy_x_test['Label']=unhealthy_y_test
    frames=[healthy_x_test, unhealthy_x_test]
    test=pd.concat(frames)
    x_test, y_test = test.iloc[:, :187], test['Label']
    
    
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)

    return [x_train_norm, x_test_norm,y_train,y_test]

def model_creation(Shape):
    model = Sequential([
    Input(shape=(Shape, 1)),
    Conv1D(128, 11, activation='relu', padding='Same'),
    MaxPool1D(pool_size=3, strides=2, padding='same'),
    Conv1D(64, 3, activation='relu', padding='Same'),
    MaxPool1D(pool_size=3, strides=2, padding='same'),
    Conv1D(64, 3, activation='relu', padding='Same'),
    MaxPool1D(pool_size=3, strides=2, padding='same'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='linear')])
    return model

def train_local_model(model, x_train_norm, y_train, x_test_norm, y_test,  epochs=5, batch_size = 32):

    model.compile(optimizer = keras.optimizers.Nadam(learning_rate=0.001),loss=SparseCategoricalCrossentropy(from_logits=True))
    model.fit(x_train_norm, y_train, epochs=5, batch_size=32)

    pred = model.predict(x_test_norm)
    pred_after_softmax = tensorflow.nn.softmax(pred)
    pred_after_softmax = pred_after_softmax.numpy() #convert to numpy to use masking further
    y_test_pred = pred_after_softmax.argmax(axis=1) #find result in every case
    print(f'Test set accuracy is {round((y_test_pred == y_test).sum() / y_test.shape[0] * 100, 2)}%')
    r = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion matrix:")
    print(r)
    precision = precision_score(y_test, y_test_pred)
    print("\nPrecision:")
    print(precision)
    print("\nRecall:")
    recall = recall_score(y_test,y_test_pred)
    print(recall)
    print("\nF1-score:")
    f1 = f1_score(y_test,y_test_pred)
    print("F1 Score:", f1)

    return model

def get_state_dict(model):
    num=1
    state_dict={}
    print("Parameters of the local model:")
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            weights = layer.get_weights()
       
        if hasattr(layer, 'bias'):
            biases = layer.bias
        l_name="layer"+str(num)
        state_dict[l_name]=[weights,biases]
        print("Layer:",l_name,"Weight:",weights," Bias",biases)
        num+=1

    

    return state_dict



def run_client():
    try:
        print("Initializing client socket")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 8888))
        print("\nConnected to the server!")

        [x_train_norm, x_test_norm, y_train, y_test] = load_data()
        print("\ndata loaded!!\n")


        # Initialize model
        model = model_creation(x_train_norm.shape[1])
        print(model.summary())
        
        model = train_local_model(model,x_train_norm, y_train, x_test_norm, y_test, epochs=5, 
                                        batch_size = 32)

        print("\nPreparing data to be sent to server\n")

        
        data_to_send =get_state_dict(model)



        # Send data to server
        pickled_data = pickle.dumps(data_to_send)

        checksum = hashlib.sha256(pickled_data).digest()
        
        pickled_data_with_checksum = pickle.dumps((checksum, pickled_data))
        # Compress the pickled data
        compressed_data = zlib.compress(pickled_data_with_checksum)
        # Transmit the compressed data
        
        client_socket.send(compressed_data)


        client_socket.close()
        print("Connection closed")
    except ConnectionRefusedError:
        print("Connection refused. Server might be offline.")
    except Exception as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    run_client()






